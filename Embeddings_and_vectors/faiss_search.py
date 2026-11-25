import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Read a text file and return its text
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Split text into chunks by paragraph (same as Day 2)
def chunk_by_paragraph(text, max_chars=500):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


text = load_txt("notes.txt")
chunks = chunk_by_paragraph(text, max_chars=400)

print("Chunks:", len(chunks))
for i, c in enumerate(chunks):
    print(f"--- chunk {i} ({len(c)} chars) ---")
    print(c[:80])
    # Step 4: turn every chunk into a vector
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = np.array(model.encode(chunks), dtype="float32")
    print("\nVectors shape:", vectors.shape)

    # Make every vector length 1, so inner product equals cosine similarity
    faiss.normalize_L2(vectors)
    print("Length of first vector:", np.linalg.norm(vectors[0]))
    # Step 5: build the FAISS index and add the vectors
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print("\nVectors in index:", index.ntotal)


    # Step 6: search function
    def search(question, k=3):
        q = np.array(model.encode([question]), dtype="float32")
        faiss.normalize_L2(q)
        k = min(k, index.ntotal)
        scores, ids = index.search(q, k)
        print("\nQuestion:", question)
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            print(f"  score {score:.3f} | chunk {i}")
            print("   ", chunks[i])


    # Ask 5 questions about the notes
    search("What is an embedding?")
    search("Why do we use overlap?")
    search("How does cosine similarity work?")
    search("How is a vector database different from SQL?")
    search("What does top-k mean?")
    search("Which library from Meta searches vectors quickly?")
    search("Why is metadata useful?")
    search("How can a computer find text with similar meaning?")

    # Observation:
    # Most questions found the right chunk at the top with a good score.
    # A question with different words also matched by meaning, which shows semantic search works.
    # One question returned a less related chunk, because a chunk can hold two topics.