// Copyright 2016 rust-tfidf Developers
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Library to calculate TF-IDF (Term Frequency - Inverse Document Frequency)
//! for generic documents. The library provides strategies to act on objects
//! that implement certain document traits (`NaiveDocument`, `ProcessedDocument`,
//! `ExpandableDocument`).
//!
//! For more information on the strategies that were implemented, check out
//! [Wikipedia](http://en.wikipedia.org/wiki/Tf%E2%80%93idf).
//!
//! # Document Types
//!
//! A document is defined as a collection of terms. The documents don't make
//! assumptions about the term types (the terms are not normalized in any way).
//!
//! These document types are of my design. The terminology isn't standard, but
//! they are fairly straight forward to understand.
//!
//!   * `NaiveDocument` - A document is 'naive' if it only knows if a term is
//!     contained within it or not, but does not know HOW MANY of the instances
//!     of the term it contains.
//!
//!   * `ProcessedDocument` - A document is 'processed' if it knows how many
//!     instances of each term is contained within it.
//!
//!   * `ExpandableDocument` - A document is 'expandable' if provides a way to
//!     access each term contained within it.
//!
//! # Example
//!
//! The most simple way to calculate the TfIdf of a document is with the default
//! implementation. Note, the library provides implementation of
//! `ProcessedDocument`, for a `Vec<(T, usize)>`.
//!
//! ```rust
//! use tfidf::{TfIdf, TfIdfDefault};
//!
//! let mut docs = Vec::new();
//! let doc1 = vec![("a", 3), ("b", 2), ("c", 4)];
//! let doc2 = vec![("a", 2), ("d", 5)];
//!
//! docs.push(doc1);
//! docs.push(doc2);
//!
//! assert_eq!(0f64, TfIdfDefault::tfidf("a", &docs[0], docs.iter()));
//! assert!(TfIdfDefault::tfidf("c", &docs[0], docs.iter()) > 0.5);
//! ```
//!
//! You can also roll your own strategies to calculate tf-idf using some strategies
//! included in the library.
//!
//! ```rust
//! use tfidf::{TfIdf, ProcessedDocument};
//! use tfidf::tf::{RawFrequencyTf};
//! use tfidf::idf::{InverseFrequencySmoothIdf};
//!
//! #[derive(Copy, Clone)] struct MyTfIdfStrategy;
//!
//! impl<T> TfIdf<T> for MyTfIdfStrategy where T : ProcessedDocument {
//!   type Tf = RawFrequencyTf;
//!   type Idf = InverseFrequencySmoothIdf;
//! }
//!
//! # let mut docs = Vec::new();
//! # let doc1 = vec![("a", 3), ("b", 2), ("c", 4)];
//! # let doc2 = vec![("a", 2), ("d", 5)];
//!
//! # docs.push(doc1);
//! # docs.push(doc2);
//!
//! assert!(MyTfIdfStrategy::tfidf("a", &docs[0], docs.iter()) > 0f64);
//! assert!(MyTfIdfStrategy::tfidf("c", &docs[0], docs.iter()) > 0f64);
//! ```

#![deny(missing_docs)]

pub use prelude::{
  Document, ExpandableDocument, Idf, NaiveDocument, NormalizationFactor, ProcessedDocument,
  SmoothingFactor, Tf, TfIdf,
};

use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

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
#[derive(Copy, Clone)]
pub struct TfIdfDefault;

impl<T> TfIdf<T> for TfIdfDefault
where
  T: ProcessedDocument,
{
  type Tf = tf::DoubleHalfNormalizationTf;
  type Idf = idf::InverseFrequencyIdf;
}

impl<T> Document for Vec<(T, usize)> {
  type Term = T;
}

impl<T> ProcessedDocument for Vec<(T, usize)>
where
  T: PartialEq,
{
  fn term_frequency<K>(&self, term: K) -> usize
  where
    K: Borrow<T>,
  {
    match self.iter().find(|&&(ref t, _)| t == term.borrow()) {
      Some(&(_, c)) => c,
      None => 0,
    }
  }

  fn max(&self) -> Option<&T> {
    match self.iter().max_by_key(|&&(_, c)| c) {
      Some(&(ref t, _)) => Some(t),
      None => None,
    }
  }
}

impl<T> Document for HashMap<T, usize> {
  type Term = T;
}

impl<T> ProcessedDocument for HashMap<T, usize>
where
  T: Eq + Hash,
{
  fn term_frequency<K>(&self, term: K) -> usize
  where
    K: Borrow<Self::Term>,
  {
    if let Some(v) = self.get(term.borrow()) {
      *v
    } else {
      0
    }
  }

  fn max(&self) -> Option<&Self::Term> {
    self.iter().max_by_key(|(_k, v)| *v).map(|(k, _v)| k)
  }
}

impl<T> Document for BTreeMap<T, usize> {
  type Term = T;
}

impl<T> ProcessedDocument for BTreeMap<T, usize>
where
  T: Ord,
{
  fn term_frequency<K>(&self, term: K) -> usize
  where
    K: Borrow<Self::Term>,
  {
    if let Some(v) = self.get(term.borrow()) {
      *v
    } else {
      0
    }
  }

  fn max(&self) -> Option<&Self::Term> {
    self.iter().max_by_key(|(_k, v)| *v).map(|(k, _v)| k)
  }
}

#[test]
fn tfidf_wiki_example_tests() {
  let mut docs = Vec::new();

  docs.push(vec![("this", 1), ("is", 1), ("a", 2), ("sample", 1)]);
  docs.push(vec![("this", 1), ("is", 1), ("another", 2), ("example", 3)]);
}
