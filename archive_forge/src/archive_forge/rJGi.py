import sqlite3
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SearchModule:
    """
    Enables detailed natural language semantic text search within embeddings.
    This module utilizes cosine similarity for semantic search and SQLite for database operations.
    """

    def __init__(self, search_db_path: str):
        """
        Initializes the search module with a connection to the search database.
        :param search_db_path: str - Path to the SQLite database file where search data and embeddings are stored.
        """
        self.db_path = search_db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def _load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Loads all embeddings from the database and converts them into a dictionary of numpy arrays.
        :return: Dict[str, np.ndarray] - A dictionary where keys are file paths and values are embeddings.
        """
        self.cursor.execute("SELECT file_path, embedding FROM embeddings")
        data = self.cursor.fetchall()
        return {
            file_path: np.frombuffer(embedding, dtype=np.float32)
            for file_path, embedding in data
        }

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates an embedding for the query text using the same model used for document embeddings.
        :param query: str - The text to generate embedding for.
        :return: np.ndarray - The generated query embedding.
        """
        # Placeholder for the actual embedding function, to be replaced with model-specific implementation
        # For example, using mlpnet_base_v2 model as in other modules
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/mlpnet-base-v2"
        )
        model = AutoModel.from_pretrained("sentence-transformers/mlpnet-base-v2")
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def perform_search(self, query: str) -> List[Dict[str, str]]:
        """
        Executes a semantic search based on the natural language query and returns relevant embedding data.
        :param query: str - The natural language query for which to find relevant documents.
        :return: List[Dict[str, str]] - A list of dictionaries containing file paths and their relevance scores.
        """
        embeddings = self._load_all_embeddings()
        query_embedding = self._generate_query_embedding(query)
        results = []
        for file_path, embedding in embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), embedding.reshape(1, -1)
            )[0][0]
            results.append({"file_path": file_path, "similarity_score": similarity})
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:10]  # Returns top 10 most relevant results

    def __del__(self):
        """
        Ensures the database connection is closed when the object is deleted.
        """
        self.connection.close()
