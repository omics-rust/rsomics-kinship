use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_kinship::open_prefix;
use std::hint::black_box;
use std::path::PathBuf;

fn bench(c: &mut Criterion) {
    let prefix = match std::env::var("RSOMICS_KINSHIP_BENCH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => return,
    };
    let bp = open_prefix(&prefix).expect("open fileset");
    let n = bp.n_samples();
    c.bench_function("king_table", |b| {
        b.iter(|| {
            for i in 1..n {
                for j in 0..i {
                    black_box(bp.pair(i, j).kinship);
                }
            }
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
