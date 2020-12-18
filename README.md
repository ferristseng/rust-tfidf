# Rust TF-IDF

[![Build Status](https://travis-ci.org/ferristseng/rust-tfidf.svg?branch=master)](https://travis-ci.org/github/ferristseng/rust-tfidf)

Library to calculate TF-IDF (Term Frequency - Inverse Document Frequency)
for generic documents. The library provides strategies to act on objects 
that implement certain document traits (`NaiveDocument`, `ProcessedDocument`,
`ExpandableDocument`).

For more information on the strategies that were implemented, check out 
[Wikipedia](http://en.wikipedia.org/wiki/Tf%E2%80%93idf).

# Document Types

A document is defined as a collection of terms. The documents don't make 
assumptions about the term types (the terms are not normalized in any way).

These document types are of my design. The terminology isn't standard, but 
they are fairly straight forward to understand.

  * `NaiveDocument` - A document is 'naive' if it only knows if a term is 
    contained within it or not, but does not know HOW MANY of the instances 
    of the term it contains.

  * `ProcessedDocument` - A document is 'processed' if it knows how many 
    instances of each term is contained within it.

  * `ExpandableDocument` - A document is 'expandable' if provides a way to 
    access each term contained within it.

# Example

The most simple way to calculate the TfIdf of a document is with the default 
implementation. Note, the library provides implementation of 
`ProcessedDocument`, for a `Vec<(T, usize)>`.

```rust
use tfidf::{TfIdf, TfIdfDefault};

let mut docs = Vec::new();
let doc1 = vec![("a", 3), ("b", 2), ("c", 4)];
let doc2 = vec![("a", 2), ("d", 5)];

docs.push(doc1);
docs.push(doc2);

assert_eq!(0f64, TfIdfDefault::tfidf("a", &docs[0], docs.iter()));
assert!(TfIdfDefault::tfidf("c", &docs[0], docs.iter()) > 0.5);
```

You can also roll your own strategies to calculate tf-idf using some strategies
included in the library. 

```rust
use tfidf::{TfIdf, ProcessedDocument};
use tfidf::tf::{RawFrequencyTf};
use tfidf::idf::{InverseFrequencySmoothIdf};

#[derive(Copy, Clone)] struct MyTfIdfStrategy;

impl<T> TfIdf<T> for MyTfIdfStrategy where T : ProcessedDocument {
  type Tf = RawFrequencyTf;
  type Idf = InverseFrequencySmoothIdf; 
}

let mut docs = Vec::new();
let doc1 = vec![("a", 3), ("b", 2), ("c", 4)];
let doc2 = vec![("a", 2), ("d", 5)];

docs.push(doc1);
docs.push(doc2);

assert!(MyTfIdfStrategy::tfidf("a", &docs[0], docs.iter()) > 0f64);
assert!(MyTfIdfStrategy::tfidf("c", &docs[0], docs.iter()) > 0f64);
```

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
