import requests
from typing import List, Dict, Any, Tuple
import uuid


class VectorDBClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the client with the API base URL."""
        self.base_url = base_url.rstrip("/")

    def check_status(self) -> Dict[str, Any]:
        """Check the status of the vector database."""
        try:
            response = requests.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to check status: {str(e)}"}

    def add_vector(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a single text and its embedding to the database."""
        try:
            payload = {"text": text}
            if metadata is not None:
                payload["metadata"] = metadata
            response = requests.post(f"{self.base_url}/add_vector", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to add vector: {str(e)}"}

    def add_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple texts and their embeddings to the database."""
        try:
            # Ensure each item has a text field and optional metadata
            for item in items:
                if "text" not in item:
                    return {"error": "Each item must have a 'text' field"}
            payload = {"items": items}
            response = requests.post(f"{self.base_url}/add_batch", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to add batch: {str(e)}"}

    def search(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar vectors given a query text."""
        try:
            payload = {"query_text": query_text, "top_k": top_k}
            response = requests.post(f"{self.base_url}/search", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to search: {str(e)}"}


# Demo usage
if __name__ == "__main__":
    # Initialize the client
    client = VectorDBClient(base_url="http://localhost:5000")

    # Check API status
    print("Checking API status:")
    status = client.check_status()
    print(status)

    # Add a single vector
    print("\nAdding a single vector:")
    result = client.add_vector(
        text="AI is transforming the world",
        metadata={"category": "AI", "source": "article"}
    )
    print(result)

    # Add a batch of vectors
    print("\nAdding a batch of vectors:")
    batch_items = [
        {"text": "Deep learning is powerful", "metadata": {"source": "book"}},
        {"text": "Coding is my passion", "metadata": {"source": "blog", "author": "John Doe"}}
    ]
    result = client.add_batch(batch_items)
    print(result)

    # Search for similar vectors
    print("\nSearching for similar vectors:")
    result = client.search(query_text="I enjoy programming and AI", top_k=3)
    print("Search results:")
    if "results" in result:
        for item in result["results"]:
            print(f"Score: {item['score']:.3f}, Metadata: {item['metadata']}")
    else:
        print(result)