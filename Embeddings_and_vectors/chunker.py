# Step 2: read a text file and return its text
def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Use the function
text = load_txt("sample.txt")

# Check the text
print("Total characters:", len(text))
print(text[:500])
# Step 3: split the text into fixed-size chunks with overlap
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


fixed = chunk_text(text, chunk_size=300, overlap=50)
print("\nFixed chunks:", len(fixed))
for i, c in enumerate(fixed[:5]):
    print(f"--- Fixed chunk {i + 1} ({len(c)} chars) ---")
    print(c)


    # Step 4: split by paragraph, and keep adding paragraphs until the chunk is full
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


    para = chunk_by_paragraph(text, max_chars=300)
    print("\nParagraph chunks:", len(para))
    print("Fixed sizes:", [len(c) for c in fixed])
    print("Paragraph sizes:", [len(c) for c in para])

    for i, c in enumerate(para[:3]):
        print(f"--- Paragraph chunk {i + 1} ({len(c)} chars) ---")
        print(c)
        # Step 5: store extra information (metadata) with every chunk
        records = []
        for i, c in enumerate(para):
            records.append({
                "text": c,
                "source": "sample.txt",
                "chunk_id": i,
            })

        print("\nFirst record:")
        print(records[0])

        # Observation: Fixed chunks cut the text at a fixed length, so a sentence
        # can be cut in the middle. Paragraph chunks keep full paragraphs, so they
        # are easier to read and better for my notes. Fixed chunks have overlap.