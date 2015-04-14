//! TF-IDF

#![deny(missing_docs)]
#![cfg_attr(test, feature(core))]

extern crate num;

pub use prelude::{Document, NaiveDocument, Tf, NormalizationFactor};

mod prelude;

/// Implementations of different weighting schemes for term frequency (tf). 
/// For more information about which ones are implemented, check the Wiki 
/// link in the crate description.
pub mod tf;

/// Implementations of different weighting schemes for inverse document 
/// frequency (IDF). For more information about which ones are implemented,
/// check the Wiki link in the crate description.
pub mod idf;

#[cfg(test)]
impl<T> Document for Vec<(T, usize)> where T : PartialEq {
  type Term = T;

  fn term_frequency(&self, term: &T) -> usize {
    match self.iter().find(|&&(ref t, _)| t == term) {
      Some(&(_, c)) => c,
      None => 0
    }
  }

  fn max(&self) -> Option<&T> {
    match self.iter().max_by(|&&(_, c)| c) {
      Some(&(ref t, _)) => Some(t),
      None => None
    }
  }
}
