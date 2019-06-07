# Word Sense Embeddings

## Problem formulation

- Given context of words surrounding a target word, model has to predict correct word sense in form of `<lemma>_<synset>`.
- Output is sense embeddings not word embeddings

---

## TODO

1. [x] Parse corpus to extract info needed for training

   1. [x] Download the high_precision (is smaller but should be more reliable and kind of double checked) dataset.
      > `wget 'http://lcl.uniroma1.it/eurosense/data/eurosense.v1.0.high-precision.tar.gz'`
   2. [x] For every sentence tag
      1. [x] choose only the text with attribute `lang=en`
      2. [x] Tokenize every sentence using `nltk.tokenize.word_tokenize(S)` which returns a list
      3. [x] Change every anchor in tokenized list (step #2) to `lemma_synset`
   3. [x] Save corpus output in text file
   4. [x] Parse Train-O-Matic dataset
      1. [x] Download dataset
         > `wget http://trainomatic.org/data/train-o-matic-data.zip`
      2. [x] parse each file
      3. [x] get context and lemma
      4. [x] replace context with `lemma_synset` pair
   5. [x] Parse SEW dataset
      1. [x] Download sew dataset
         > `wget http://lcl.uniroma1.it/sew/data/sew_conservative.tar.gz`
      - only first 50 million sentences

   - In all datasets, all the characters were lowered, and punctuation characters were removed.

2. [x] Restrict sense embeddings only to senses in wordnet

   1. [x] file `bn2wn_mapping.txt` which contains the mapping from BabelNet synset ids to WordNet offset ids

3. [x] Train WORD2VEC model to create Sense embeddings

   1. [x] Save the output in form of `embeddings.vec`

4. [x] Test sense embeddings in word similarity text
   1. [x] Download "wordsim353" `wget 'http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip'` and use combined.tab version
   2. [x] cosine similarity
      - [x] still needs to be finalized, no clear plan to implement it yet.
5. [x] Visualize Results using T-SNE 2D & 3D

   1. [x] Sketchy implementation of the plot method
   2. [x] Test implementation

6. [x] Pipeline

- Best results acquired:
  - weighted cosine similarity = 0.5954 on 45 million sentences, approximately
