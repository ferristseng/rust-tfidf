/// A naive document with a simple function stating whether or not a 
/// term exists in the document or not. The document is naive , which 
/// means the frequencies of each term has yet to be determined. This 
/// type of document is useful for only some TF weighting schemes.
pub trait NaiveDocument {
  /// The type of term that the document consists of.
  type Term; 

  /// Returns if a (non-normalized) term exists within the document. 
  fn term_exists(&self, term: &Self::Term) -> bool;
}

/// A document where the frequencies of each term is already calculated.
pub trait Document {
  /// The type of term that the document consists of.
  type Term;

  /// Returns the number of times a (non-normalized) term exists 
  /// within the document.
  fn term_frequency(&self, term: &Self::Term) -> usize;

  /// Returns the term with the highest frequency, or tied for the highest
  /// frequency.
  fn max(&self) -> Option<&Self::Term>;
}

/// A document that can be expanded to a collection of terms.
pub trait ExpandableDocument {
  /// The type of term that the document consists of.
  type Term;

  /// The type of iterator that this implementor returns.
  type TermIterator : Iterator<Item = Self::Term>;

  /// An iterator over the terms in the document.
  fn terms(&self) -> Self::TermIterator;
}

impl<D, T> NaiveDocument for D where D : Document<Term = T> {
  type Term = T;

  #[inline] fn term_exists(&self, term: &Self::Term) -> bool { 
    self.term_frequency(term) > 0 
  }
}

/// A strategy to calculate a weighted or unweighted term frequency (tf)
/// score of a term from a document.
pub trait Tf<T> where T : NaiveDocument {
  /// Returns the weighted or unweighted term frequency (tf) for a single 
  /// term within a document.
  fn tf(term: &T::Term, doc: &T) -> f64;
}

/// A strategy to calculate a weighted or unweighted inverse document frequency
/// (idf) for a single term within a corpus of documents.
pub trait Idf<T> where T : NaiveDocument {
  /// Returns the weighted or unweighted inverse document frequency (idf) 
  /// for a single term within a corpus of documents.
  fn idf<'a, I>(term: &T::Term, docs: I) -> f64 where I : Iterator<Item = &'a T>;
}

/// A strategy that uses a normalization factor.
pub trait NormalizationFactor {
  /// Returns a normalization factor.
  fn factor() -> f64;
}
