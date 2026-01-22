import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
from embedding_storage import EmbeddingStorage
from data_analysis import DataAnalysis
from nlp_nlu_processor import NLPNLUProcessor
from file_processor import FileProcessor


class KnowledgeGraph:
    """
    Constructs and manages a dynamic, interactive, scalable, and efficient knowledge graph based on embeddings.
    This module uses NetworkX for in-memory graph operations and Neo4j for persistent graph storage.
    It integrates with EmbeddingStorage, DataAnalysis, NLPNLUProcessor, and FileProcessor to utilize all generated data.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        embedding_db_path: str,
        analysis_db_path: str,
        nlp_db_path: str,
        folder_path: str,
    ):
        """
        Initializes the knowledge graph module with a connection to the graph database and other necessary components.
        :param uri: str - URI for the Neo4j database.
        :param user: str - Username for the Neo4j database.
        :param password: str - Password for the Neo4j database.
        :param embedding_db_path: str - Path to the SQLite database file where embeddings are stored.
        :param analysis_db_path: str - Path to the SQLite database file for storing analysis results.
        :param nlp_db_path: str - Path to the SQLite database file for storing NLP processed embeddings.
        :param folder_path: str - Path to the folder containing documents to be processed and embedded.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.graph = nx.DiGraph()
        self.embedding_storage = EmbeddingStorage(embedding_db_path, folder_path)
        self.data_analysis = DataAnalysis(analysis_db_path, embedding_db_path)
        self.nlp_processor = NLPNLUProcessor(embedding_db_path, nlp_db_path)
        self.file_processor = FileProcessor(folder_path)
        logging.basicConfig(level=logging.INFO)

    def update_graph(self, analysis_results: List[Dict[str, Any]]) -> None:
        """
        Updates the knowledge graph with new analysis results, integrating data into Neo4j and NetworkX graph.
        :param analysis_results: List[Dict[str, Any]] - Results from data analysis to be integrated into the graph.
        """
        with self.driver.session() as session:
            for result in analysis_results:
                # Convert embeddings to proper format before merging
                embedding_bytes = np.array(result["embedding"]).tobytes()
                session.run(
                    "MERGE (a:Document {file_path: $file_path}) "
                    "ON CREATE SET a.embedding = $embedding",
                    file_path=result["file_path"],
                    embedding=embedding_bytes,
                )
                for relation in result["relations"]:
                    session.run(
                        "MATCH (a:Document {file_path: $file_path}) "
                        "MERGE (b:Document {file_path: $related_file}) "
                        "MERGE (a)-[:RELATED_TO {type: $relation_type}]->(b)",
                        file_path=result["file_path"],
                        related_file=relation["file_path"],
                        relation_type=relation["type"],
                    )
                self.graph.add_node(
                    result["file_path"],
                    embedding=np.frombuffer(embedding_bytes, dtype=np.float32),
                )
                for relation in result["relations"]:
                    self.graph.add_edge(
                        result["file_path"],
                        relation["file_path"],
                        type=relation["type"],
                    )

    def display_graph(self) -> None:
        """
        Renders the interactive knowledge graph using matplotlib for visualization.
        """
        pos = nx.spring_layout(self.graph, seed=42)  # for consistent layout
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="skyblue",
            edge_color="k",
            node_size=700,
            font_size=10,
        )
        plt.show()

    def __del__(self):
        """
        Closes the database connection when the object is deleted.
        """
        self.driver.close()
