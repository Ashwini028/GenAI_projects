# compare_search.py
# Goal: compare keyword search (BM25), semantic search (embeddings)
# and hybrid search (BM25 + embeddings merged with RRF)

import re  # regex module, used to get words from text
from rank_bm25 import BM25Okapi  # BM25 class for keyword scoring
from sentence_transformers import SentenceTransformer  # embedding model
import numpy as np  # arrays and sorting


def load_txt(filename):
    # Open the file for reading and close it automatically at the end
    with open(filename, "r", encoding="utf-8") as f:
        # Return the whole text as one string
        return f.read()


def chunk_by_paragraph(text, max_chars=500):
    # Split the text at blank lines into paragraphs
    paragraphs = text.split("\n\n")
    chunks = []  # final list of chunks
    current = ""  # the chunk we are building now
    for para in paragraphs:
        para = para.strip()  # remove extra spaces at both ends
        if not para:
            continue  # skip empty paragraphs
        # If the paragraph fits in the current chunk, join it
        if len(current) + len(para) + 1 <= max_chars:
            current = (current + " " + para).strip()
        else:
            # It does not fit: save the finished chunk
            if current:
                chunks.append(current)
            # Start a new chunk with this paragraph
            current = para
    # Save the last chunk too
    if current:
        chunks.append(current)
    return chunks


# Collect the chunks of both files in one list
all_chunks = []
for filename in ["sample.txt", "notes.txt"]:
    text = load_txt(filename)  # read the file
    for chunk in chunk_by_paragraph(text, max_chars=400):
        all_chunks.append(chunk)  # add each chunk to the list

print("Total chunks:", len(all_chunks))


# ---------- Keyword search (BM25) ----------

def tokenize(text):
    # Lowercase the text and keep only words (no punctuation like ? or ,)
    return re.findall(r"\w+", text.lower())


# Turn every chunk into a list of words
tokenized_chunks = [tokenize(chunk) for chunk in all_chunks]
# Give all chunks to BM25. It calculates TF and IDF by itself
bm25 = BM25Okapi(tokenized_chunks)


def keyword_search(question, k=3):
    # BM25 score of every chunk for this question
    scores = bm25.get_scores(tokenize(question))
    # argsort gives indexes from low to high score, [::-1] reverses it,
    # [:k] keeps the top k
    top_ids = np.argsort(scores)[::-1][:k]
    # Return (chunk index, score) pairs
    return [(int(i), float(scores[i])) for i in top_ids]


def show_results(title, results):
    # Print the results neatly with rank, score and the start of the chunk
    print(title)
    for rank, (i, score) in enumerate(results, start=1):
        print(f"  {rank}. score={score:.4f} chunk={i}: {all_chunks[i][:100]}")


# ---------- Semantic search (embeddings) ----------

# Load the same embedding model we used on Day 1-4
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert all chunks to vectors. normalize_embeddings=True makes every
# vector length 1, so a dot product becomes the cosine similarity
chunk_embeddings = model.encode(all_chunks, normalize_embeddings=True)


def semantic_search(question, k=3):
    # Convert the question to a vector in the same way
    q_emb = model.encode(question, normalize_embeddings=True)
    # Dot product of every chunk vector with the question vector.
    # Bigger score means closer meaning
    scores = chunk_embeddings @ q_emb
    # Top k indexes, biggest score first
    top_ids = np.argsort(scores)[::-1][:k]
    # Return (chunk index, score) pairs
    return [(int(i), float(scores[i])) for i in top_ids]


# ---------- Hybrid search (RRF = Reciprocal Rank Fusion) ----------

def hybrid_search(question, k=3, pool=5):
    # Take the top "pool" chunk ids from each method (we only need the ids)
    keyword_ids = [i for i, _ in keyword_search(question, k=pool)]
    semantic_ids = [i for i, _ in semantic_search(question, k=pool)]

    # Dictionary: chunk id -> total RRF score
    rrf_scores = {}
    for ranked_list in [keyword_ids, semantic_ids]:
        # rank is 1 for the best chunk, 2 for the next one, and so on
        for rank, chunk_id in enumerate(ranked_list, start=1):
            # RRF formula: add 1 / (60 + rank) from every list
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (60 + rank)

    # Sort by RRF score, biggest first, and keep the top k
    best = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return best


# Test with one question
# ---------- Compare all three methods on many questions ----------

# Mix of questions: some use the exact words of the notes,
# some use different words but the same meaning
questions = [
    "What is overlap?",  # exact word
    "How do I keep context between pieces of text?",  # same meaning, no word "overlap"
    "How is a vector database different from SQL?",  # mixed
    "Why use metadata?",  # exact word
    "Which method finds text with similar meaning?",  # meaning based
]

for question in questions:
    print("=" * 60)  # a line to separate the questions
    print("QUESTION:", question)
    # k=2 shows only the top 2 results of each method, to keep output short
    show_results("KEYWORD (BM25)", keyword_search(question, k=2))
    show_results("SEMANTIC (embeddings)", semantic_search(question, k=2))
    show_results("HYBRID (RRF)", hybrid_search(question, k=2))

    # Observation:
    # 1. For exact-word questions (What is overlap?), keyword, semantic and
    #    hybrid search all found the right chunk.
    # 2. For "How do I keep context between pieces of text?", keyword search
    #    failed. It matched mostly common words and returned the embeddings
    #    chunk. Semantic search found the overlap chunk by meaning, and hybrid
    #    search also put the correct chunk first.
    # 3. For "Why use metadata?", all methods gave low scores (BM25 about 1.0,
    #    semantic about 0.47). My notes probably have no good answer. In RAG, a
    #    minimum score can be used to say "I don't know".
    # 4. sample.txt and notes.txt have almost the same paragraphs, so chunks
    #    0/2 and 1/3 are duplicates and fill the top-k. Fix: remove duplicates.
    # Conclusion: keyword is good for exact words, semantic is good for
    # meaning, and hybrid is the safest choice for a RAG app.