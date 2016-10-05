// Copyright 2016 rust-tfidf Developers
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::borrow::Borrow;

use prelude::{ProcessedDocument, NaiveDocument, Tf, NormalizationFactor};

/// Binary weighting scheme for TF. If the document contains the term, returns 1,
/// otherwise returns 0.
#[derive(Copy, Clone)]
pub struct BinaryTf;

impl<T> Tf<T> for BinaryTf where T: NaiveDocument
{
  #[inline]
  fn tf<K>(term: K, doc: &T) -> f64
    where K: Borrow<T::Term>
  {
    if doc.term_exists(term) {
      1f64
    } else {
      0f64
    }
  }
}

/// Raw frequency weighting scheme for TF. Returns the number of times a term occurs
/// in the document.
#[derive(Copy, Clone)]
pub struct RawFrequencyTf(f64);

impl<T> Tf<T> for RawFrequencyTf where T: ProcessedDocument
{
  #[inline]
  fn tf<K>(term: K, doc: &T) -> f64
    where K: Borrow<T::Term>
  {
    doc.term_frequency(term) as f64
  }
}

/// Log normalized weighting scheme for TF. Computes `1 + log (f)` where `f` is the
/// frequency of the term in the document.
#[derive(Copy, Clone)]
pub struct LogNormalizationTf;

impl<T> Tf<T> for LogNormalizationTf where T: ProcessedDocument
{
  #[inline]
  fn tf<K>(term: K, doc: &T) -> f64
    where K: Borrow<T::Term>
  {
    1f64 + (doc.term_frequency(term) as f64).ln()
  }
}

/// Double normalized weighting scheme for TF based on a factor, `K`.
///
/// # Example
///
/// To implement a custom Tf strategy, where the `K` factor is constant:
///
/// ```rust
/// use tfidf::{Tf, NormalizationFactor};
/// use tfidf::tf::{DoubleKNormalizationTf};
///
/// struct DoubleThirdNormalizationTf;
///
/// impl NormalizationFactor for DoubleThirdNormalizationTf {
///   fn factor() -> f64 { 0.3f64 }
/// }
///
/// impl DoubleKNormalizationTf for DoubleThirdNormalizationTf { }
/// ```
pub trait DoubleKNormalizationTf : NormalizationFactor { }

impl<T, S> Tf<T> for S
  where S: DoubleKNormalizationTf,
        T: ProcessedDocument
{
  #[inline]
  fn tf<K>(term: K, doc: &T) -> f64
    where K: Borrow<T::Term>
  {
    let max = match doc.max() {
      Some(m) => doc.term_frequency(m) as f64,
      None => 1f64,
    };

    // K + ((1 - K) * (f / max f))
    S::factor() + ((1f64 - S::factor()) * ((doc.term_frequency(term) as f64) / max))
  }
}

/// Double normalized weighting scheme for TF based on a factor, `K = 0.5`.
#[derive(Copy, Clone)]
pub struct DoubleHalfNormalizationTf;

impl NormalizationFactor for DoubleHalfNormalizationTf {
  #[inline]
  fn factor() -> f64 {
    0.5f64
  }
}

impl DoubleKNormalizationTf for DoubleHalfNormalizationTf {}
