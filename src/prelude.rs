use std::borrow::Borrow;

/// A body of terms.
pub trait Document {
  /// The type of term that the document consists of.
  type Term;
}

/// A naive document with a simple function stating whether or not a
/// term exists in the document or not. The document is naive , which
/// means the frequencies of each term has yet to be determined. This
/// type of document is useful for only some TF weighting schemes.
pub trait NaiveDocument : Document {
  /// Returns if a (non-normalized) term exists within the document.
  fn term_exists<K>(&self, term: K) -> bool where K: Borrow<Self::Term>;
}

/// A document where the frequencies of each term is already calculated.
pub trait ProcessedDocument : Document {
  /// Returns the number of times a (non-normalized) term exists
  /// within the document.
  fn term_frequency<K>(&self, term: K) -> usize where K: Borrow<Self::Term>;

  /// Returns the term with the highest frequency, or tied for the highest
  /// frequency.
  fn max(&self) -> Option<&Self::Term>;
}

/// A document that can be expanded to a collection of terms.
pub trait ExpandableDocument<'a> : Document
  where <Self as Document>::Term : 'a
{
  /// The type of iterator that this implementor returns.
  type TermIterator : Iterator<Item = &'a Self::Term>;

  /// An iterator over the terms in the document.
  fn terms(&self) -> Self::TermIterator;
}

impl<D, T> NaiveDocument for D where D: ProcessedDocument<Term = T>
{
  #[inline]
  fn term_exists<K>(&self, term: K) -> bool
    where K: Borrow<T>
  {
    self.term_frequency(term) > 0
  }
}

/// A strategy to calculate a weighted or unweighted term frequency (tf)
/// score of a term from a document.
pub trait Tf<T> where T : NaiveDocument {
  /// Returns the weighted or unweighted term frequency (tf) for a single
  /// term within a document.
  fn tf<K>(term: K, doc: &T) -> f64 where K: Borrow<T::Term>;
}

/// A strategy to calculate a weighted or unweighted inverse document frequency
/// (idf) for a single term within a corpus of documents.
pub trait Idf<T> where T : NaiveDocument {
  /// Returns the weighted or unweighted inverse document frequency (idf)
  /// for a single term within a corpus of documents.
  fn idf<'a, I, K>(term: K, docs: I) -> f64
    where I: Iterator<Item = &'a T>,
          K: Borrow<T::Term>,
          T: 'a;
}

/// A strategy that uses a normalization factor.
pub trait NormalizationFactor {
  /// Returns a normalization factor.
  fn factor() -> f64;
}

/// A strategy that uses a smoothing factor.
pub trait SmoothingFactor {
  /// Returns a smoothing factor.
  fn factor() -> f64;
}

/// Trait to create a strategy to calculate a tf-idf.
pub trait TfIdf<T> where T : NaiveDocument {
  /// The tf weighting scheme.
  type Tf : Tf<T>;

  /// The idf weighting scheme.
  type Idf : Idf<T>;

  /// Calculates the tf-idf using the two weighting schemes chosen.
  fn tfidf<'a, K, I>(term: K, doc: &T, docs: I) -> f64
    where I: Iterator<Item = &'a T>,
          K: Borrow<T::Term>,
          T: 'a
  {
    Self::Tf::tf(term.borrow(), doc) * Self::Idf::idf(term.borrow(), docs)
  }
}
