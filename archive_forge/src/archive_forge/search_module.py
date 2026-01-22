import sqlite3
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from embedding_storage import EmbeddingStorage
from nlp_nlu_processor import NLPNLUProcessor
from data_analysis import DataAnalysis
from knowledge_graph import KnowledgeGraph


class SearchModule:
    """
    Enables detailed natural language semantic text search within embeddings across multiple databases.
    This module utilizes cosine similarity for semantic search and integrates with other modules for comprehensive data handling.
    """

    def __init__(
        self,
        search_db_path: str,
        embedding_db_path: str,
        nlp_db_path: str,
        analysis_db_path: str,
    ):
        """
        Initializes the search module with connections to multiple databases and integrates with other modules.
        :param search_db_path: str - Path to the SQLite database file where search data and embeddings are stored.
        :param embedding_db_path: str - Path to the SQLite database file where embeddings are stored.
        :param nlp_db_path: str - Path to the SQLite database file for storing NLP processed embeddings.
        :param analysis_db_path: str - Path to the SQLite database file for storing analysis results.
        """
        self.db_path = search_db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.embedding_storage = EmbeddingStorage(embedding_db_path, None)
        self.nlp_processor = NLPNLUProcessor(embedding_db_path, nlp_db_path)
        self.data_analysis = DataAnalysis(analysis_db_path, embedding_db_path)
        self.knowledge_graph = KnowledgeGraph(
            "bolt://localhost:7687",
            "neo4j",
            "password",
            embedding_db_path,
            analysis_db_path,
            nlp_db_path,
            None,
        )

    def _load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Loads all embeddings from the database and converts them into a dictionary of numpy arrays.
        :return: Dict[str, np.ndarray] - A dictionary where keys are file paths and values are embeddings.
        """
        return self.embedding_storage.load_embeddings()

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates an embedding for the query text using the NLP/NLU processor.
        :param query: str - The text to generate embedding for.
        :return: np.ndarray - The generated query embedding.
        """
        return self.nlp_processor.generate_embedding(query)

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
        self.knowledge_graph.__del__()
