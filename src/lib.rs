//! Storage, logical declarations, and scalar algebra for symbolic and concrete
//! relations.
#![warn(missing_docs)]

pub mod util;

pub mod ast;
pub mod backend;
pub mod bdd;
mod cache;
pub mod context;
pub mod scalar;
pub mod sparse;
