import chromadb
from sentence_transformers import SentenceTransformer


# Read a text file
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Split text into chunks by paragraph
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


# Try it on one file
text = load_txt("notes.txt")
chunks = chunk_by_paragraph(text, max_chars=400)
print("Chunks in notes.txt:", len(chunks))
# Step 4: create a persistent Chroma client and a collection
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="my_notes",
    metadata={"hnsw:space": "cosine"},
)
print("Chunks already saved:", collection.count())
# Step 5: add the chunks of both files to Chroma, with metadata
files = ["sample.txt", "notes.txt"]
for filename in files:
    text = load_txt(filename)
    chunks = chunk_by_paragraph(text, max_chars=400)
    embeddings = model.encode(chunks).tolist()
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

print("Total chunks in Chroma:", collection.count())
# Step 6: search function, with an optional filter on the source file
def search(question, k=3, source=None):
    q_emb = model.encode([question]).tolist()
    if source:
        results = collection.query(
            query_embeddings=q_emb,
            n_results=k,
            where={"source": source},
        )
    else:
        results = collection.query(query_embeddings=q_emb, n_results=k)

    print("\nQuestion:", question, "| filter:", source)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        print(f"  distance {dist:.3f} | {meta['source']} chunk {meta['chunk_id']}")
        print("   ", doc)


search("What is overlap?")
search("What is overlap?", source="sample.txt")
search("How is a vector database different from SQL?")
search("Why use metadata?", source="notes.txt")
search("What is chunk overlap used for?")



# Observation:
# Without a filter, the results came from both files. With the filter
# source="sample.txt", all results came only from that file, so the
# metadata filter works. Only 2 results came back because sample.txt has only 2 chunks.
# The SQL question got a very small distance (0.279), because its chunk has one topic.
# The overlap question had a larger distance (0.541), because its chunk mixed 3 topics.
# On the second run, "Chunks already saved" was 8, so the data is persisted.
# Chroma returns the text and metadata directly, so no separate chunks list is needed like in FAISS.