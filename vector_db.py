import numpy as np
from typing import List, Dict, Any, Tuple


class SimpleVectorDB:
    def __init__(self, dim: int):
        """Initialize vector database with specified embedding dimension."""
        self.dim = dim
        self.vectors = []
        self.metadata = []

    def add_vector(self, vector: np.ndarray, meta: Dict[str, Any]) -> None:
        """Add a vector with associated metadata to the database."""
        if vector.shape != (self.dim,):
            raise ValueError(f"Vector dimension must be {self.dim}, got {vector.shape}")
        self.vectors.append(vector)
        self.metadata.append(meta)

    def add_batch(self, vectors: List[np.ndarray], metas: List[Dict[str, Any]]) -> None:
        """Add multiple vectors and their metadata in a batch."""
        if len(vectors) != len(metas):
            raise ValueError("Number of vectors must match number of metadata entries")
        for vec, meta in zip(vectors, metas):
            self.add_vector(vec, meta)

    def cosine_similarity(self, query: np.ndarray) -> List[Tuple[float, Dict[str, Any]]]:
        """Find similar vectors using cosine similarity, return sorted (score, metadata) pairs."""
        if not self.vectors:
            return []

        query = query.reshape(-1)
        if query.shape != (self.dim,):
            raise ValueError(f"Query dimension must be {self.dim}, got {query.shape}")

        # Convert vectors to numpy array for efficient computation
        vectors = np.array(self.vectors)

        # Compute cosine similarity: (a · b) / (||a|| ||b||)
        dot_products = np.dot(vectors, query)
        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = dot_products / (vector_norms * query_norm + 1e-10)  # Avoid division by zero

        # Pair similarities with metadata and sort by similarity (descending)
        results = [(sim, self.metadata[i]) for i, sim in enumerate(similarities)]
        return sorted(results, key=lambda x: x[0], reverse=True)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Return top_k most similar vectors with their metadata."""
        return self.cosine_similarity(query)[:top_k]


# Demo usage
if __name__ == "__main__":
    # Initialize database with 3D vectors (small dimension for demo)
    db = SimpleVectorDB(dim=3)

    # Sample data: text and their "embeddings" (random for demo)
    sample_data = [
        ("I love to code in Python", np.array([0.1, 0.5, 0.2])),
        ("Machine learning is fun", np.array([0.3, 0.1, 0.4])),
        ("Python is great for AI", np.array([0.2, 0.4, 0.3])),
        ("Data science is exciting", np.array([0.4, 0.2, 0.1]))
    ]

    # Add sample data to the database
    for text, vector in sample_data:
        db.add_vector(vector, {"text": text})

    # Query vector (random example)
    query_vector = np.array([0.15, 0.45, 0.25])

    # Perform similarity search
    results = db.search(query_vector, top_k=2)

    # Print results
    print("Query results:")
    for score, meta in results:
        print(f"Score: {score:.3f}, Text: {meta['text']}")