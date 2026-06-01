use std::path::{Path, PathBuf};
use std::process::Command;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn bin() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push(if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    });
    p.push("rsomics-kinship");
    if p.exists() {
        return p;
    }
    // workspace / CARGO_TARGET_DIR layouts
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-kinship"))
}

fn run_ours(prefix: &Path, out_prefix: &Path) {
    let status = Command::new(bin())
        .arg(prefix)
        .arg("-o")
        .arg(out_prefix)
        .status()
        .expect("spawn rsomics-kinship");
    assert!(status.success(), "rsomics-kinship exited non-zero");
}

/// Always-on: ours against the committed plink2-captured `.kin0`.
#[test]
fn matches_committed_golden() {
    let gd = golden_dir();
    let scratch = tempfile::tempdir().expect("tempdir");
    let out = scratch.path().join("ours");
    run_ours(&gd.join("gold"), &out);

    let got = std::fs::read_to_string(out.with_extension("kin0")).unwrap();
    let want = std::fs::read_to_string(gd.join("gold_king.kin0")).unwrap();
    assert_eq!(got, want, "ours diverged from the committed plink2 golden");
}

/// Live differential: regenerate the golden with the real plink2 binary and
/// diff byte-for-byte. Loud-skips when plink2 is not installed so CI (which
/// has no plink2) stays green while a developer with it gets the full check.
#[test]
fn matches_live_plink2() {
    let plink2 = std::env::var("PLINK2").unwrap_or_else(|_| "plink2".to_string());
    if Command::new(&plink2).arg("--version").output().is_err() {
        eprintln!("SKIP matches_live_plink2: `{plink2}` not found (set PLINK2=/path/to/plink2)");
        return;
    }

    let gd = golden_dir();
    let scratch = tempfile::tempdir().expect("tempdir");
    let p2_out = scratch.path().join("p2");
    let status = Command::new(&plink2)
        .arg("--bfile")
        .arg(gd.join("gold"))
        .arg("--make-king-table")
        .arg("--out")
        .arg(&p2_out)
        .status()
        .expect("spawn plink2");
    assert!(status.success(), "plink2 exited non-zero");

    let ours_out = scratch.path().join("ours");
    run_ours(&gd.join("gold"), &ours_out);

    let got = std::fs::read_to_string(ours_out.with_extension("kin0")).unwrap();
    let p2 = std::fs::read_to_string(p2_out.with_extension("kin0")).unwrap();
    assert_eq!(got, p2, "ours diverged from live plink2 --make-king-table");
}
