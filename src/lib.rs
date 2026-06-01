mod fileset;
mod format;
mod gfmt;
mod king;
mod simd;

pub use fileset::{Bitplanes, open, open_prefix};
pub use format::write_table;
pub use king::{PairResult, kinship};
