use anyhow::{Context, bail};
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct Sample {
    pub fid: String,
    pub iid: String,
}

/// A PLINK1 binary fileset transposed into per-sample bitplanes for the KING
/// popcount kernel. Each sample owns four `words`-long `u64` bit arrays over
/// variants: `het` (genotype 1), `hom_ref` (genotype 2 = two A1 alleles),
/// `hom_alt` (genotype 0), and `nonmiss` (any called genotype). `het_total` is
/// each sample's het count, exact whenever the dataset has no missing calls.
pub struct Bitplanes {
    pub samples: Vec<Sample>,
    pub n_variants: usize,
    pub words: usize,
    pub het: Vec<u64>,
    pub hom_ref: Vec<u64>,
    pub hom_alt: Vec<u64>,
    pub nonmiss: Vec<u64>,
    pub het_total: Vec<u32>,
    pub has_missing: bool,
    pub use_simd: bool,
}

impl Bitplanes {
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    #[inline]
    pub fn plane<'a>(&self, planes: &'a [u64], s: usize) -> &'a [u64] {
        &planes[s * self.words..(s + 1) * self.words]
    }
}

pub fn open_prefix(prefix: &Path) -> anyhow::Result<Bitplanes> {
    open(
        &prefix.with_extension("bed"),
        &prefix.with_extension("bim"),
        &prefix.with_extension("fam"),
    )
}

pub fn open(bed: &Path, bim: &Path, fam: &Path) -> anyhow::Result<Bitplanes> {
    let n_variants = count_lines(bim)?;
    let samples = parse_fam(fam)?;
    let n_samples = samples.len();
    if n_samples == 0 {
        bail!("no samples in {}", fam.display());
    }
    let bytes_per_variant = n_samples.div_ceil(4);

    let file = File::open(bed).with_context(|| format!("open {}", bed.display()))?;
    let map = unsafe { Mmap::map(&file)? };
    if map.len() < 3 || map[0] != 0x6c || map[1] != 0x1b {
        bail!("bad .bed magic in {}", bed.display());
    }
    match map[2] {
        0x01 => {}
        0x00 => bail!(
            "sample-major .bed ({}); rerun PLINK with --make-bed",
            bed.display()
        ),
        other => bail!("unknown .bed mode byte {other:#x} in {}", bed.display()),
    }
    let expected = 3 + bytes_per_variant * n_variants;
    if map.len() != expected {
        bail!(
            ".bed size mismatch: {} bytes, expected {} for {}×{}",
            map.len(),
            expected,
            n_variants,
            n_samples
        );
    }

    let words = n_variants.div_ceil(64);
    let mut het = vec![0u64; n_samples * words];
    let mut hom_ref = vec![0u64; n_samples * words];
    let mut hom_alt = vec![0u64; n_samples * words];
    let mut nonmiss = vec![0u64; n_samples * words];

    for v in 0..n_variants {
        let row = &map[3 + v * bytes_per_variant..3 + (v + 1) * bytes_per_variant];
        let word = v / 64;
        let bit = 1u64 << (v % 64);
        for s in 0..n_samples {
            let code = (row[s >> 2] >> ((s & 3) << 1)) & 0b11;
            let off = s * words + word;
            match code {
                0b00 => hom_ref[off] |= bit,
                0b10 => het[off] |= bit,
                0b11 => hom_alt[off] |= bit,
                _ => continue,
            }
            nonmiss[off] |= bit;
        }
    }

    let het_total: Vec<u32> = (0..n_samples)
        .map(|s| {
            het[s * words..(s + 1) * words]
                .iter()
                .map(|w| w.count_ones())
                .sum()
        })
        .collect();
    let called: u64 = nonmiss.iter().map(|w| w.count_ones() as u64).sum();
    let has_missing = called != (n_samples * n_variants) as u64;

    Ok(Bitplanes {
        samples,
        n_variants,
        words,
        het,
        hom_ref,
        hom_alt,
        nonmiss,
        het_total,
        has_missing,
        use_simd: crate::simd::supported(),
    })
}

fn count_lines(path: &Path) -> anyhow::Result<usize> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut n = 0;
    for line in BufReader::new(f).lines() {
        if !line?.trim_end().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

fn parse_fam(path: &Path) -> anyhow::Result<Vec<Sample>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (lineno, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        let t = line.trim_end();
        if t.is_empty() {
            continue;
        }
        let mut it = t.split_whitespace();
        match (it.next(), it.next()) {
            (Some(fid), Some(iid)) => out.push(Sample {
                fid: fid.to_string(),
                iid: iid.to_string(),
            }),
            _ => bail!(
                "{}:{}: expected at least FID IID",
                path.display(),
                lineno + 1
            ),
        }
    }
    Ok(out)
}
