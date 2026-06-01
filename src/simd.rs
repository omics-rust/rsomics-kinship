//! AVX2 Harley-Seal popcount for the KING no-missing hot loop. Two reductions
//! run in lockstep: `hethet = Σ popcount(het_i & het_j)` and
//! `ibs0 = Σ popcount((hr_i & ha_j) | (ha_i & hr_j))`. The carry-save-adder
//! tree (Mula & Lemire, "Faster Population Counts Using AVX2") folds sixteen
//! 256-bit vectors per iteration into a handful of byte-shuffle popcounts,
//! pushing the popcount rate well below scalar `popcnt`.
//!
//! Selected at runtime via `is_x86_feature_detected!`, so a stock
//! `cargo install` binary gets it on any AVX2 host with no build-time flags.

#[cfg(target_arch = "x86_64")]
pub fn supported() -> bool {
    is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn supported() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
pub fn pair_counts(
    het_i: &[u64],
    het_j: &[u64],
    hr_i: &[u64],
    hr_j: &[u64],
    ha_i: &[u64],
    ha_j: &[u64],
) -> (u64, u64) {
    unsafe { pair_counts_avx2(het_i, het_j, hr_i, hr_j, ha_i, ha_j) }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn csa(a: __m256i, b: __m256i, c: __m256i) -> (__m256i, __m256i) {
    let u = _mm256_xor_si256(a, b);
    let low = _mm256_xor_si256(u, c);
    let high = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    (high, low)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn popcnt_vec(v: __m256i) -> __m256i {
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    let bytes = _mm256_add_epi8(
        _mm256_shuffle_epi8(lookup, lo),
        _mm256_shuffle_epi8(lookup, hi),
    );
    _mm256_sad_epu8(bytes, _mm256_setzero_si256())
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256i) -> u64 {
    let mut lanes = [0u64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v) };
    lanes[0] + lanes[1] + lanes[2] + lanes[3]
}

/// One Harley-Seal accumulator over a per-iteration supplier of 256-bit
/// vectors. Folds 16 vectors per round; the supplier returns the k-th fused
/// vector for the current 16-vector window.
#[cfg(target_arch = "x86_64")]
struct Hs {
    total: __m256i,
    ones: __m256i,
    twos: __m256i,
    fours: __m256i,
    eights: __m256i,
}

#[cfg(target_arch = "x86_64")]
impl Hs {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn new() -> Self {
        Hs {
            total: _mm256_setzero_si256(),
            ones: _mm256_setzero_si256(),
            twos: _mm256_setzero_si256(),
            fours: _mm256_setzero_si256(),
            eights: _mm256_setzero_si256(),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn round(&mut self, d: &[__m256i; 16]) {
        unsafe {
            let (twos_a, ones) = csa(self.ones, d[0], d[1]);
            self.ones = ones;
            let (twos_b, ones) = csa(self.ones, d[2], d[3]);
            self.ones = ones;
            let (fours_a, twos) = csa(self.twos, twos_a, twos_b);
            self.twos = twos;
            let (twos_a, ones) = csa(self.ones, d[4], d[5]);
            self.ones = ones;
            let (twos_b, ones) = csa(self.ones, d[6], d[7]);
            self.ones = ones;
            let (fours_b, twos) = csa(self.twos, twos_a, twos_b);
            self.twos = twos;
            let (eights_a, fours) = csa(self.fours, fours_a, fours_b);
            self.fours = fours;

            let (twos_a, ones) = csa(self.ones, d[8], d[9]);
            self.ones = ones;
            let (twos_b, ones) = csa(self.ones, d[10], d[11]);
            self.ones = ones;
            let (fours_a, twos) = csa(self.twos, twos_a, twos_b);
            self.twos = twos;
            let (twos_a, ones) = csa(self.ones, d[12], d[13]);
            self.ones = ones;
            let (twos_b, ones) = csa(self.ones, d[14], d[15]);
            self.ones = ones;
            let (fours_b, twos) = csa(self.twos, twos_a, twos_b);
            self.twos = twos;
            let (eights_b, fours) = csa(self.fours, fours_a, fours_b);
            self.fours = fours;
            let (sixteens, eights) = csa(self.eights, eights_a, eights_b);
            self.eights = eights;

            self.total = _mm256_add_epi64(self.total, popcnt_vec(sixteens));
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn finish(self) -> u64 {
        unsafe {
            let mut acc = _mm256_slli_epi64(self.total, 4);
            acc = _mm256_add_epi64(acc, _mm256_slli_epi64(popcnt_vec(self.eights), 3));
            acc = _mm256_add_epi64(acc, _mm256_slli_epi64(popcnt_vec(self.fours), 2));
            acc = _mm256_add_epi64(acc, _mm256_slli_epi64(popcnt_vec(self.twos), 1));
            acc = _mm256_add_epi64(acc, popcnt_vec(self.ones));
            hsum(acc)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pair_counts_avx2(
    het_i: &[u64],
    het_j: &[u64],
    hr_i: &[u64],
    hr_j: &[u64],
    ha_i: &[u64],
    ha_j: &[u64],
) -> (u64, u64) {
    unsafe {
        let words = het_i.len();
        let rounds = (words / 4) / 16;

        let ph_i = het_i.as_ptr() as *const __m256i;
        let ph_j = het_j.as_ptr() as *const __m256i;
        let pr_i = hr_i.as_ptr() as *const __m256i;
        let pr_j = hr_j.as_ptr() as *const __m256i;
        let pa_i = ha_i.as_ptr() as *const __m256i;
        let pa_j = ha_j.as_ptr() as *const __m256i;

        let mut hh = Hs::new();
        let mut ib = Hs::new();

        for r in 0..rounds {
            let base = r * 16;
            let mut hvec = [_mm256_setzero_si256(); 16];
            let mut bvec = [_mm256_setzero_si256(); 16];
            for k in 0..16 {
                let idx = base + k;
                hvec[k] = _mm256_and_si256(
                    _mm256_loadu_si256(ph_i.add(idx)),
                    _mm256_loadu_si256(ph_j.add(idx)),
                );
                bvec[k] = _mm256_or_si256(
                    _mm256_and_si256(
                        _mm256_loadu_si256(pr_i.add(idx)),
                        _mm256_loadu_si256(pa_j.add(idx)),
                    ),
                    _mm256_and_si256(
                        _mm256_loadu_si256(pa_i.add(idx)),
                        _mm256_loadu_si256(pr_j.add(idx)),
                    ),
                );
            }
            hh.round(&hvec);
            ib.round(&bvec);
        }

        let mut hh_n = hh.finish();
        let mut ib_n = ib.finish();

        for w in (rounds * 64)..words {
            hh_n += (het_i[w] & het_j[w]).count_ones() as u64;
            ib_n += ((hr_i[w] & ha_j[w]) | (ha_i[w] & hr_j[w])).count_ones() as u64;
        }
        (hh_n, ib_n)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn scalar(
        hi: &[u64],
        hj: &[u64],
        ri: &[u64],
        rj: &[u64],
        ai: &[u64],
        aj: &[u64],
    ) -> (u64, u64) {
        let mut hh = 0;
        let mut ib = 0;
        for w in 0..hi.len() {
            hh += (hi[w] & hj[w]).count_ones() as u64;
            ib += ((ri[w] & aj[w]) | (ai[w] & rj[w])).count_ones() as u64;
        }
        (hh, ib)
    }

    #[test]
    fn avx2_matches_scalar() {
        if !supported() {
            return;
        }
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // exercise sizes around the 64-word round boundary plus a tail
        for words in [64, 67, 128, 195, 1000, 1563] {
            let mk = |rng: &mut dyn FnMut() -> u64| (0..words).map(|_| rng()).collect::<Vec<_>>();
            let hi = mk(&mut rng);
            let hj = mk(&mut rng);
            let ri = mk(&mut rng);
            let rj = mk(&mut rng);
            let ai = mk(&mut rng);
            let aj = mk(&mut rng);
            let got = pair_counts(&hi, &hj, &ri, &rj, &ai, &aj);
            let want = scalar(&hi, &hj, &ri, &rj, &ai, &aj);
            assert_eq!(got, want, "words={words}");
        }
    }
}
