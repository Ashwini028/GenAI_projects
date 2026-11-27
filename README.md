 # Generative AI Learning Projects

 My learning projects for Generative AI, built week by week.

 ## Week 2: Day 1 :Embeddings and Vector Search

 Folder: `Embeddings_and_vectors`

 - `cosine_basics.py`: cosine similarity written from scratch with NumPy.
 - `embed_sentences.py`: converts 5 sentences into embeddings with
   sentence-transformers (all-MiniLM-L6-v2, 384 dimensions) and
   prints a similarity table.

 ### What I learned
 - An embedding is a list of numbers that represents the meaning of text.
 - Sentences with similar meaning get a high cosine score even with
   different words ("I love pizza" and "Pizza is my favourite food": 0.867).
 - Unrelated sentences get a score close to 0.

 ## How to run

     pip install numpy sentence-transformers
     python Embeddings_and_vectors/embed_sentences.py



## Day 2: Document Loading and Chunking

- `chunker.py`: loads a text file, splits it into chunks in two ways
  (fixed-size with overlap, and by paragraph), and stores each chunk
  with metadata (source file and chunk id).
- `sample.txt`: small text file used to test the chunkers.

### What I learned
- Fixed-size chunks can cut a sentence in the middle.
- Paragraph chunks keep full paragraphs, so ideas stay together.
- Overlap repeats a little text between chunks, so context is not lost.
- Metadata (source, chunk id) helps to show sources and filter later.

To run: open the folder `Embeddings_and_vectors` and run `python chunker.py`.





## Day 3: Vector Database and FAISS

- `faiss_search.py`: splits `notes.txt` into chunks, creates embeddings with
  sentence-transformers, normalizes them, stores them in a FAISS index
  (IndexFlatIP), and answers questions with the top 3 matching chunks.
- `notes.txt`: small notes file used for the search.

### What I learned
- A vector database stores embeddings and finds the closest ones to a query.
- FAISS is a fast search library. It stores only vectors, so I keep a
- 
  separate list of chunks to get the text.
- Normalized vectors with an inner product index give cosine similarity.
- Exact search checks every vector. ANN search is faster but approximate.

To run: open the folder `Embeddings_and_vectors` and run `python faiss_search.py`.





## Week 2 Day 4: ChromaDB search
- What it does: Splits text files into chunks, stores them in ChromaDB with metadata (source, chunk_id), and finds the most similar chunks for a question.
- Tech: Python, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), cosine distance, metadata filter, persistent storage.