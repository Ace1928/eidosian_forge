from typing import Dict, List, Optional, Tuple
import spacy
from langdetect import detect
from IntelligentVectorDB.embedding_storage import (
    EmbeddingStorage,
)  # Importing the EmbeddingStorage class for embedding operations
import numpy as np
import sqlite3


class NLPNLUProcessor:
    """
    Provides advanced NLP and NLU functionalities to process and understand text data,
    integrating natural language understanding capabilities. This class utilizes the spaCy library
    for NLP tasks and langdetect for language detection, supporting multiple languages with specific
    models loaded dynamically based on language detection. Additionally, it integrates with the
    EmbeddingStorage class to handle embeddings data for further analysis and ensures that text data
    is standardized across modules by encoding and storing processed text as embeddings in a dedicated
    NLP analysis database.

    Attributes:
        nlp_models (Dict[str, spacy.lang]): Dictionary of loaded spaCy language models.
        embedding_storage (EmbeddingStorage): Instance of EmbeddingStorage to manage embeddings.
        nlp_db_path (str): Path to the SQLite database for storing NLP processed embeddings.
    """

    def __init__(self, embedding_db_path: str, nlp_db_path: str):
        """
        Initializes the NLP/NLU processing capabilities. Loads spaCy language models that are necessary
        for processing different languages and tasks, initially for English and Spanish. Also initializes
        the embedding storage with the specified database path for embeddings and sets up a database
        connection for storing NLP processed embeddings.

        :param embedding_db_path: str - Path to the SQLite database file where embeddings are stored.
        :param nlp_db_path: str - Path to the SQLite database file for storing NLP processed embeddings.
        """
        self.nlp_models = {
            "en": spacy.load("en_core_web_md"),  # English medium model
            "es": spacy.load("es_core_news_md"),  # Spanish medium model
        }
        self.embedding_storage = EmbeddingStorage(
            embedding_db_path, None
        )  # Initialize without a folder path
        self.nlp_db_path = nlp_db_path
        self.nlp_db_connection = sqlite3.connect(self.nlp_db_path)
        self.nlp_db_cursor = self.nlp_db_connection.cursor()
        self._create_nlp_tables()

    def _create_nlp_tables(self) -> None:
        """
        Creates the necessary tables in the NLP database if they do not already exist.
        """
        self.nlp_db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS nlp_embeddings (
                id INTEGER PRIMARY KEY,
                text TEXT UNIQUE,
                embedding BLOB
            )
            """
        )
        self.nlp_db_connection.commit()

    def process_text(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Processes text to detect its language, and then extract and return advanced linguistic features including
        tokens, entities, and sentence structures. It also handles the storage of embeddings for the processed text
        and ensures that the text is standardized across modules by storing the processed embeddings in the NLP database.

        :param text: str - Text to be processed.
        :return: Dict[str, List[Dict[str, str]]] - A dictionary containing tokens, entities, and sentences.
        """
        if not self._is_text_processed(text):
            language = self.language_detection(text)
            nlp = self.nlp_models.get(
                language, self.nlp_models["en"]
            )  # Default to English if model not found
            processed_data = self._enhance_text_data(text, nlp)
            embedding = self.embedding_storage._generate_embeddings(text)
            self._store_nlp_embeddings(text, embedding)
            self.embedding_storage.store_embeddings(
                text
            )  # Store embeddings of the processed text
            return processed_data
        else:
            return self._retrieve_processed_data(text)

    def _is_text_processed(self, text: str) -> bool:
        """
        Checks if the text has already been processed and stored in the NLP database.

        :param text: str - Text to check.
        :return: bool - True if the text is already processed, False otherwise.
        """
        self.nlp_db_cursor.execute(
            "SELECT text FROM nlp_embeddings WHERE text = ?", (text,)
        )
        return self.nlp_db_cursor.fetchone() is not None

    def _store_nlp_embeddings(self, text: str, embedding: np.ndarray) -> None:
        """
        Stores the processed text embeddings in the NLP database.

        :param text: str - Text that was processed.
        :param embedding: np.ndarray - Embedding of the processed text.
        """
        self.nlp_db_cursor.execute(
            "INSERT OR REPLACE INTO nlp_embeddings (text, embedding) VALUES (?, ?)",
            (text, embedding.tobytes()),
        )
        self.nlp_db_connection.commit()

    def _retrieve_processed_data(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Retrieves the processed data for the text from the NLP database if it has been previously processed.

        :param text: str - Text to retrieve processed data for.
        :return: Dict[str, List[Dict[str, str]]] - Processed features including tokens, entities, sentences.
        """
        self.nlp_db_cursor.execute(
            "SELECT embedding FROM nlp_embeddings WHERE text = ?", (text,)
        )
        embedding_bytes = self.nlp_db_cursor.fetchone()[0]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        # Assuming a method to convert embeddings back to text data exists
        processed_data = self._convert_embedding_to_data(embedding)
        return processed_data

    def _convert_embedding_to_data(
        self, embedding: np.ndarray
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Converts embeddings back to text data. This is a placeholder for the actual implementation.

        :param embedding: np.ndarray - Embedding to convert.
        :return: Dict[str, List[Dict[str, str]]] - Simulated processed data.
        """
        # Simulated return for demonstration purposes
        return {"tokens": [], "entities": [], "sentences": []}

    def language_detection(self, text: str) -> str:
        """
        Determines the language of the given text using the langdetect library.

        :param text: str - Text whose language is to be detected.
        :return: str - ISO 639-1 language code (e.g., 'en' for English, 'es' for Spanish).
        """
        return detect(text)

    def named_entity_recognition(self, text: str) -> List[Dict[str, str]]:
        """
        Identifies and classifies named entities in the text using the detected language model.

        :param text: str - Text to be analyzed for named entities.
        :return: List[Dict[str, str]] - A list of entities with text and type.
        """
        language = self.language_detection(text)
        nlp = self.nlp_models.get(
            language, self.nlp_models["en"]
        )  # Default to English if model not found
        doc = nlp(text)
        return [{"text": entity.text, "type": entity.label_} for entity in doc.ents]
