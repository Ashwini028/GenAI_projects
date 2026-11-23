# Import the model loader and NumPy
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a small embedding model (it downloads on the first run)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 5 sentences: 3 about food, 1 about train, 1 about weather
sentences = [
    "I love pizza",
    "Pizza is my favourite food",
    "I enjoy eating pasta",
    "The train is late",
    "It is raining today",
]

# Convert every sentence into a vector of 384 numbers
vectors = model.encode(sentences)

# Shape should be (5, 384): 5 sentences, 384 numbers each
print("Shape:", vectors.shape)
# Same cosine function as before
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Compare every sentence with every other sentence
print("\nSimilarity table:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cosine_similarity(vectors[i], vectors[j])
        print(f"{sentences[i]}  <->  {sentences[j]}  =  {score:.3f}")