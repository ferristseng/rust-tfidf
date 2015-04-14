//! TF-IDF

#![deny(missing_docs)]
#![cfg_attr(test, feature(core))]

extern crate num;

pub use prelude::{Document, NaiveDocument, ExpandableDocument, ProcessedDocument, 
  Tf, NormalizationFactor, SmoothingFactor, TfIdf};

#[cfg(test)] use std::borrow::Borrow;

mod prelude;

/// Implementations of different weighting schemes for term frequency (tf). 
/// For more information about which ones are implemented, check the Wiki 
/// link in the crate description.
pub mod tf;

/// Implementations of different weighting schemes for inverse document 
/// frequency (IDF). For more information about which ones are implemented,
/// check the Wiki link in the crate description.
pub mod idf;

/// Default scheme for calculating tf-idf.
#[derive(Copy, Clone)] pub struct TfIdfDefault;

impl<T> TfIdf<T> for TfIdfDefault where T : ProcessedDocument {
  type Tf = tf::DoubleHalfNormalizationTf;
  type Idf = idf::InverseFrequencyIdf;
}

#[cfg(test)]
impl<T> Document for Vec<(T, usize)> {
  type Term = T;
}

#[cfg(test)]
impl<T> ProcessedDocument for Vec<(T, usize)> where T : PartialEq {
  fn term_frequency<K>(&self, term: K) -> usize where K : Borrow<T> {
    match self.iter().find(|&&(ref t, _)| t == term.borrow()) {
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
