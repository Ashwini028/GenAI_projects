# Step 1: import the NumPy library. "np" is its short name.
import numpy as np


# Step 2: make a function that takes two vectors a and b.
def cosine_similarity(a, b):
    # dot product: multiply matching numbers and add them
    dot_product = np.dot(a, b)

    # length (norm) of each vector
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)

    # cosine = dot product divided by (length of a * length of b)
    return dot_product / (length_a * length_b)


# Step 3: test the function with 4 small examples.

# Test 1: slightly similar vectors
a = np.array([1, 0])
b = np.array([1, 1])
print("Test 1:", cosine_similarity(a, b))

# Test 2: same direction, b is just longer
a = np.array([1, 2, 3])
b = np.array([2, 4, 6])
print("Test 2:", cosine_similarity(a, b))

# Test 3: completely different directions (90 degrees)
a = np.array([1, 0])
b = np.array([0, 1])
print("Test 3:", cosine_similarity(a, b))

# Test 4: opposite directions
a = np.array([1, 2])
b = np.array([-1, -2])
print("Test 4:", cosine_similarity(a, b))