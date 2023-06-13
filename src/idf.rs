// Copyright 2016 rust-tfidf Developers
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use prelude::{ExpandableDocument, Idf, NaiveDocument, ProcessedDocument, SmoothingFactor};

/// Unary weighting scheme for IDF. If the corpus contains a document with the
/// term, returns 1, otherwise returns 0.
#[derive(Copy, Clone)]
pub struct UnaryIdf;

impl<T> Idf<T> for UnaryIdf
where
  T: NaiveDocument,
{
  #[inline]
  fn idf<'a, I, K>(term: K, docs: I) -> f64
  where
    I: Iterator<Item = &'a T>,
    K: Borrow<T::Term>,
    T: 'a,
  {
    docs.fold(0f64, |_, d| {
      if d.term_exists(term.borrow()) {
        1f64
      } else {
        0f64
      }
    })
  }
}

/// Inverse frequency weighting scheme for IDF with a smoothing factor. Used
/// internally as a marker trait.
pub trait InverseFrequencySmoothedIdfStrategy: SmoothingFactor {}

impl<S, T> Idf<T> for S
where
  S: InverseFrequencySmoothedIdfStrategy,
  T: NaiveDocument,
{
  #[inline]
  fn idf<'a, I, K>(term: K, docs: I) -> f64
  where
    I: Iterator<Item = &'a T>,
    K: Borrow<T::Term>,
    T: 'a,
  {
    let (num_docs, ttl_docs) = docs.fold((0f64, 0f64), |(n, t), d| {
      (
        if d.term_exists(term.borrow()) {
          n + 1f64
        } else {
          n
        },
        t + 1f64,
      )
    });
    (S::factor() + (ttl_docs / num_docs)).ln()
  }
}

/// Inverse frequency weighting scheme for IDF. Computes `log (N / nt)` where `N`
/// is the number of documents, and `nt` is the number of times a term appears in
/// the corpus of documents.
#[derive(Copy, Clone)]
pub struct InverseFrequencyIdf;

impl SmoothingFactor for InverseFrequencyIdf {
  fn factor() -> f64 {
    0f64
  }
}

impl InverseFrequencySmoothedIdfStrategy for InverseFrequencyIdf {}

/// Inverse frequency weighting scheme for IDF. Computes `log (1 + (N / nt))`.
#[derive(Copy, Clone)]
pub struct InverseFrequencySmoothIdf;

impl SmoothingFactor for InverseFrequencySmoothIdf {
  fn factor() -> f64 {
    1f64
  }
}

impl InverseFrequencySmoothedIdfStrategy for InverseFrequencySmoothIdf {}

/// Inverse frequency weighting scheme for IDF. Compute `log (1 + (max nt / nt))`
/// where `nt` is the number of times a term appears in the corpus, and `max nt`
/// returns the most number of times any term appears in the corpus.
#[derive(Copy, Clone)]
pub struct InverseFrequencyMaxIdf;

impl<'l, T, E> Idf<T> for InverseFrequencyMaxIdf
where
  T: ProcessedDocument<Term = E> + ExpandableDocument<'l>,
  E: Hash + Eq + 'l,
{
  #[inline]
  fn idf<'a, I, K>(term: K, docs: I) -> f64
  where
    I: Iterator<Item = &'a T>,
    K: Borrow<T::Term>,
    T: 'a,
  {
    let mut counts: HashMap<&T::Term, usize> = HashMap::new();
    let num_docs = docs.fold(0, |n, d| {
      for t in d.terms() {
        counts.insert(t, 0);
      }

      if d.term_exists(term.borrow()) {
        n + 1
      } else {
        n
      }
    });
    let max = *counts.values().max().unwrap_or(&1);

    (1f64 + (max as f64 / num_docs as f64)).ln()
  }
}

#[test]
fn idf_wiki_example_tests() {
  let mut docs = Vec::new();

  docs.push(vec![("this", 1), ("is", 1), ("a", 2), ("sample", 1)]);
  docs.push(vec![("this", 1), ("is", 1), ("another", 2), ("example", 3)]);

  assert_eq!(UnaryIdf::idf("this", docs.iter()), 1f64);
  assert_eq!(InverseFrequencyIdf::idf("this", docs.iter()), 0f64);
}

#[test]
fn idf_wiki_example_tests_hashmap() {
  let mut docs: Vec<std::collections::HashMap<&'static str, usize>> = Vec::new();

  docs.push(
    vec![("this", 1), ("is", 1), ("a", 2), ("sample", 1)]
      .into_iter()
      .collect(),
  );
  docs.push(
    vec![("this", 1), ("is", 1), ("another", 2), ("example", 3)]
      .into_iter()
      .collect(),
  );

  assert_eq!(UnaryIdf::idf("this", docs.iter()), 1f64);
  assert_eq!(InverseFrequencyIdf::idf("this", docs.iter()), 0f64);
}

#[test]
fn idf_wiki_example_tests_btreemap() {
  let mut docs: Vec<std::collections::BTreeMap<&'static str, usize>> = Vec::new();

  docs.push(
    vec![("this", 1), ("is", 1), ("a", 2), ("sample", 1)]
      .into_iter()
      .collect(),
  );
  docs.push(
    vec![("this", 1), ("is", 1), ("another", 2), ("example", 3)]
      .into_iter()
      .collect(),
  );

  assert_eq!(UnaryIdf::idf("this", docs.iter()), 1f64);
  assert_eq!(InverseFrequencyIdf::idf("this", docs.iter()), 0f64);
}
