use crate::fileset::Bitplanes;
use crate::gfmt::{g, g_ratio};
use crate::king::PairResult;
use rayon::prelude::*;
use std::fmt::Write as _;
use std::io::Write;

/// Whether any sample carries a non-trivial family ID, matching plink2's rule
/// for emitting the FID columns.
fn has_fid(bp: &Bitplanes) -> bool {
    bp.samples.iter().any(|s| s.fid != "0")
}

pub fn write_table<W: Write>(
    bp: &Bitplanes,
    min_kinship: Option<f64>,
    w: &mut W,
) -> std::io::Result<()> {
    let fid = has_fid(bp);
    if fid {
        writeln!(w, "#FID1\tIID1\tFID2\tIID2\tNSNP\tHETHET\tIBS0\tKINSHIP")?;
    } else {
        writeln!(w, "#IID1\tIID2\tNSNP\tHETHET\tIBS0\tKINSHIP")?;
    }

    let n = bp.n_samples();
    let rows: Vec<PairResult> = (1..n)
        .into_par_iter()
        .flat_map_iter(|i| (0..i).map(move |j| (i, j)))
        .map(|(i, j)| bp.pair(i, j))
        .filter(|r| min_kinship.is_none_or(|t| r.kinship >= t))
        .collect();

    let mut buf = String::with_capacity(96);
    for r in &rows {
        let s1 = &bp.samples[r.i];
        let s2 = &bp.samples[r.j];
        buf.clear();
        if fid {
            let _ = write!(buf, "{}\t{}\t{}\t{}\t", s1.fid, s1.iid, s2.fid, s2.iid);
        } else {
            let _ = write!(buf, "{}\t{}\t", s1.iid, s2.iid);
        }
        let _ = write!(
            buf,
            "{}\t{}\t{}\t{}",
            r.nsnp,
            g_ratio(r.hethet as u64, r.nsnp as u64, 6),
            g_ratio(r.ibs0 as u64, r.nsnp as u64, 6),
            g(r.kinship, 6)
        );
        writeln!(w, "{buf}")?;
    }
    Ok(())
}
