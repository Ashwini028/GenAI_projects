 # Generative AI Learning Projects

 My learning projects for Generative AI, built week by week.

 ## Week 2: Embeddings and Vector Search

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