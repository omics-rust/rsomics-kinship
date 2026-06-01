use clap::Parser;
use rsomics_kinship::{open, open_prefix, write_table};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "rsomics-kinship",
    about = "KING-robust kinship coefficients between sample pairs (plink2 --make-king-table)",
    version
)]
struct Cli {
    /// Path prefix for the .bed/.bim/.fam fileset (without extension).
    #[arg(value_name = "PREFIX")]
    bfile: Option<PathBuf>,

    /// Path to the .bed file (requires --bim and --fam).
    #[arg(long, requires_all = ["bim", "fam"], conflicts_with = "bfile")]
    bed: Option<PathBuf>,
    #[arg(long)]
    bim: Option<PathBuf>,
    #[arg(long)]
    fam: Option<PathBuf>,

    /// Emit the king-table form (FID1 IID1 FID2 IID2 ... KINSHIP). On by
    /// default; present for symmetry with plink2's flag.
    #[arg(long)]
    table: bool,

    /// Drop pairs whose kinship is below this threshold (plink2
    /// --king-table-filter).
    #[arg(long, value_name = "MIN")]
    king_table_filter: Option<f64>,

    /// Write to <OUT>.kin0 instead of stdout (plink2 --out).
    #[arg(short = 'o', long)]
    out: Option<PathBuf>,

    /// Worker threads for the pairwise pass.
    #[arg(short = 't', long, default_value_t = num_cpus())]
    threads: usize,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads.max(1))
        .build_global()
        .ok();

    let bp = match (&cli.bfile, &cli.bed) {
        (Some(prefix), None) => open_prefix(prefix)?,
        (None, Some(bed)) => open(bed, cli.bim.as_ref().unwrap(), cli.fam.as_ref().unwrap())?,
        (Some(_), Some(_)) => unreachable!("clap conflicts_with"),
        (None, None) => anyhow::bail!("expected PREFIX or --bed/--bim/--fam"),
    };

    match cli.out {
        Some(prefix) => {
            let mut w =
                BufWriter::with_capacity(1 << 20, File::create(prefix.with_extension("kin0"))?);
            write_table(&bp, cli.king_table_filter, &mut w)?;
            w.flush()?;
        }
        None => {
            let stdout = io::stdout();
            let mut w = BufWriter::with_capacity(1 << 20, stdout.lock());
            write_table(&bp, cli.king_table_filter, &mut w)?;
            w.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_definition_is_valid() {
        <Cli as clap::CommandFactory>::command().debug_assert();
    }
}
