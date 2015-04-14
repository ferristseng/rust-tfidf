use prelude::{NaiveDocument, Document, ExpandableDocument, Idf};

/// Unary weighting scheme for IDF. If the corpus contains a document with the 
/// term, returns 1, otherwise returns 0.
#[derive(Copy, Clone)] pub struct UnaryIdf;

impl<T> Idf<T> for UnaryIdf where T : NaiveDocument {
  #[inline] fn idf<'a, I>(term: &T::Term, docs: I) -> f64 where I : Iterator<Item = &'a T> {
    docs.fold(0f64, |_, d| if d.term_exists(term) { 1f64 } else { 0f64 })
  }
}

/// Inverse frequency weighting scheme for IDF. Computes `log (N / nt)` where `N` is the 
/// number of documents, and `nt` is the number of times a term appears in the corpus of 
/// documents.
#[derive(Copy, Clone)] pub struct InverseFrequencyIdf;

impl<T> Idf<T> for InverseFrequencyIdf where T : Document {
  #[inline] fn idf<'a, I>(term: &T::Term, docs: I) -> f64 where I : Iterator<Item = &'a T> {
    let (num_docs, ttl_docs) = docs.fold(
      (0f64, 0f64), 
      |(n, t), d| (if d.term_exists(term) { n + 1f64 } else { n }, t + 1f64));
    (ttl_docs as f64 / num_docs as f64).ln()
  }
}

#[test]
fn tfidf_wiki_example_tests() {
  let mut docs = Vec::new();

  docs.push(vec![("this", 1), ("is", 1), ("a", 2), ("sample", 1)]);
  docs.push(vec![("this", 1), ("is", 1), ("another", 2), ("example", 3)]);

  assert_eq!(InverseFrequencyIdf::idf(&"this", docs.iter()), 0f64);
}
