# rsomics-kinship

KING-robust kinship coefficients between every sample pair in a PLINK1
`.bed/.bim/.fam` fileset — the estimator computed by `plink2 --make-king-table`.

```
rsomics-kinship data            # reads data.{bed,bim,fam}, table to stdout
rsomics-kinship data -o out     # writes out.kin0
rsomics-kinship data --king-table-filter 0.0884 -o out
```

Output is the `.kin0` table: `#FID1 IID1 FID2 IID2 NSNP HETHET IBS0 KINSHIP`
(the FID columns are dropped when every family ID is `0`, matching plink2).

The kinship estimator is the between-family KING-robust form, normalised so
duplicate samples score 0.5:

```
phi = (2*HetHet - 4*IBS0 - Het1 - Het2 + 2*min(Het1,Het2)) / (4*min(Het1,Het2))
```

where counts run over the SNPs both samples genotype (pairwise non-missing).
The kernel transposes the genotype matrix into per-sample bitplanes so each
pair reduces to AND + `popcount` over 64-bit words, and the O(n²) pair loop is
spread over `rayon` workers.

## Origin

Independent Rust implementation of the KING-robust kinship estimator (as
exposed by `plink2 --make-king-table`) based on:

- Manichaikul et al. 2010, *Robust relationship inference in genome-wide
  association studies*, Bioinformatics 26(22):2867-2873,
  doi:10.1093/bioinformatics/btq559 (the KING-robust estimator, Eq. 11)
- Chang et al. 2015 (PLINK 2.0, doi:10.1186/s13742-015-0047-8)
- The PLINK 2.0 `.kin0` file-format spec
  (https://www.cog-genomics.org/plink/2.0/formats#kin0)
- Black-box behaviour testing against the plink2 binary

No source code from the GPL plink2 upstream was used as reference during
implementation. Test fixtures are generated independently.

License: MIT OR Apache-2.0.
Upstream credit: PLINK 2.0 (Christopher Chang et al., GPLv3),
KING (Wei-Min Chen et al.).
