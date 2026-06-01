use crate::fileset::Bitplanes;

pub struct PairResult {
    pub i: usize,
    pub j: usize,
    pub nsnp: u32,
    pub hethet: u32,
    pub ibs0: u32,
    pub het1: u32,
    pub het2: u32,
    pub kinship: f64,
}

/// KING-robust between-family kinship (Manichaikul et al. 2010, Eq. 11):
///
///   phi = [2*HetHet - 4*IBS0 - Het1 - Het2 + 2*min(Het1,Het2)] / (4*min(Het1,Het2))
///
/// Het1/Het2 count heterozygous genotypes over the SNPs both samples genotype;
/// IBS0 counts opposite-homozygote sites; HetHet counts shared hets.
#[inline]
pub fn kinship(hethet: u32, ibs0: u32, het1: u32, het2: u32) -> f64 {
    let minhet = het1.min(het2) as f64;
    if minhet == 0.0 {
        return f64::NAN;
    }
    (2.0 * hethet as f64 - 4.0 * ibs0 as f64 - het1 as f64 - het2 as f64 + 2.0 * minhet)
        / (4.0 * minhet)
}

impl Bitplanes {
    pub fn pair(&self, i: usize, j: usize) -> PairResult {
        let het_i = self.plane(&self.het, i);
        let het_j = self.plane(&self.het, j);
        let hr_i = self.plane(&self.hom_ref, i);
        let hr_j = self.plane(&self.hom_ref, j);
        let ha_i = self.plane(&self.hom_alt, i);
        let ha_j = self.plane(&self.hom_alt, j);

        let mut hethet = 0u64;
        let mut ibs0 = 0u64;

        let (nsnp, het1, het2);
        if !self.has_missing {
            if self.use_simd {
                let (hh, ib) = crate::simd::pair_counts(het_i, het_j, hr_i, hr_j, ha_i, ha_j);
                hethet = hh;
                ibs0 = ib;
            } else {
                for w in 0..self.words {
                    hethet += (het_i[w] & het_j[w]).count_ones() as u64;
                    ibs0 += ((hr_i[w] & ha_j[w]) | (ha_i[w] & hr_j[w])).count_ones() as u64;
                }
            }
            nsnp = self.n_variants as u32;
            het1 = self.het_total[i];
            het2 = self.het_total[j];
        } else {
            let nm_i = self.plane(&self.nonmiss, i);
            let nm_j = self.plane(&self.nonmiss, j);
            let mut snp = 0u64;
            let mut h1 = 0u64;
            let mut h2 = 0u64;
            for w in 0..self.words {
                let nm = nm_i[w] & nm_j[w];
                snp += nm.count_ones() as u64;
                hethet += (het_i[w] & het_j[w]).count_ones() as u64;
                ibs0 += ((hr_i[w] & ha_j[w]) | (ha_i[w] & hr_j[w])).count_ones() as u64;
                h1 += (het_i[w] & nm).count_ones() as u64;
                h2 += (het_j[w] & nm).count_ones() as u64;
            }
            nsnp = snp as u32;
            het1 = h1 as u32;
            het2 = h2 as u32;
        }

        let hethet = hethet as u32;
        let ibs0 = ibs0 as u32;
        PairResult {
            i,
            j,
            nsnp,
            hethet,
            ibs0,
            het1,
            het2,
            kinship: kinship(hethet, ibs0, het1, het2),
        }
    }
}
