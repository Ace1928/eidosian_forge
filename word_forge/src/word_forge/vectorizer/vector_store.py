"""
Vector Store module for Word Forge.

This module provides storage and retrieval capabilities for vector embeddings,
enabling semantic search and similarity operations across linguistic data.
It supports multiple storage backends and embedding models with configurable
parameters for balancing performance and accuracy.

Architecture:
    ┌─────────────────┐
    │   VectorStore   │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    │    Components   │
    └─────────────────┘
    ┌─────┬─────┬─────┐
    │Model│Index│Query│
    └─────┴─────┴─────┘
"""

from __future__ import annotations
from eidosian_core import eidosian

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,  # Added TypeAlias
    TypedDict,
    Union,
    cast,  # Keep cast for external library interactions
    overload,
)

_VECTOR_IMPORT_ERROR: Optional[Exception]
try:  # Optional heavy dependencies
    import chromadb
    from sentence_transformers import SentenceTransformer
except Exception as import_error:  # pragma: no cover - allow running without chromadb
    chromadb = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    _VECTOR_IMPORT_ERROR = import_error
else:
    _VECTOR_IMPORT_ERROR = None

try:  # Optional FAISS dependency for fallback persistence
    import faiss  # type: ignore
except (
    Exception
) as faiss_error:  # pragma: no cover - allow running without faiss initially
    faiss = None  # type: ignore
    _FAISS_IMPORT_ERROR = faiss_error
else:
    _FAISS_IMPORT_ERROR = None

import numpy as np
from numpy.typing import NDArray

from word_forge.config import config
from word_forge.database.database_manager import DatabaseError, DBManager, WordEntryDict
from word_forge.emotion.emotion_manager import EmotionManager

# Type definitions for clarity and constraint
VectorID: TypeAlias = Union[
    int, str
]  # Unique identifier for vectors (compatible with ChromaDB)
SearchResultList: TypeAlias = List[
    Tuple[VectorID, float]
]  # (id, distance) pairs - Renamed from SearchResult
ContentType: TypeAlias = Literal[
    "word", "definition", "example", "message", "conversation"
]
EmbeddingList: TypeAlias = List[float]  # Type for ChromaDB's embedding format
QueryType: TypeAlias = Literal["search", "definition", "similarity"]
TemplateDict: TypeAlias = Dict[str, Optional[str]]
WordID: TypeAlias = Union[int, str]  # ID can be int or str for ChromaDB compatibility


class VectorMetadata(TypedDict, total=False):
    """
    Metadata associated with stored vectors.

    Structured information that accompanies vector embeddings to provide
    context and support filtering operations during search.

    Attributes:
        original_id: Source entity identifier
        content_type: Category of content this vector represents
        term: Word or phrase if this vector represents lexical content
        definition: Meaning of the term if applicable
        speaker: Person who created this content if from a conversation
        emotion_valence: Emotional valence score if sentiment analyzed
        emotion_arousal: Emotional arousal intensity if sentiment analyzed
        emotion_label: Text label for the dominant emotion
        conversation_id: Parent conversation identifier for message content
        timestamp: When this content was created or processed
        language: Language code for the content
    """

    original_id: int
    content_type: ContentType
    term: Optional[str]
    definition: Optional[str]
    speaker: Optional[str]
    emotion_valence: Optional[float]
    emotion_arousal: Optional[float]
    emotion_label: Optional[str]
    conversation_id: Optional[int]
    timestamp: Optional[float]
    language: Optional[str]


class SearchResultDict(TypedDict):
    """
    Type definition for search result items.

    Structured format for returning vector search results with all
    relevant metadata and context.

    Attributes:
        id: int - Unique identifier of the matching item
        distance: float - Semantic distance from query (lower is better)
        metadata: Optional[VectorMetadata] - Associated metadata for the match
        text: Optional[str] - Raw text content if available
    """

    id: int
    distance: float
    metadata: Optional[VectorMetadata]
    text: Optional[str]


class VectorItem(TypedDict):
    """
    Type definition for vector items to be stored.

    Structured format for vector data with associated metadata and text.

    Attributes:
        id: VectorID - Unique identifier for the vector
        text: str - Text content associated with the vector
        metadata: VectorMetadata - Metadata for filtering and context
        vector: NDArray[np.float32] - Vector embedding
    """

    id: VectorID
    text: str
    metadata: VectorMetadata
    vector: NDArray[np.float32]


class InstructionTemplate(TypedDict):
    """
    Type definition for instruction template.

    Format specification for instruction-tuned language models
    that require specific prompting patterns.

    Attributes:
        task: str - Description of the task to perform
        query_prefix: str - Text to prepend to query inputs
        document_prefix: Optional[str] - Optional text to prepend to documents
    """

    task: str
    query_prefix: str
    document_prefix: Optional[str]


class ChromaCollection(Protocol):
    """
    Protocol defining required ChromaDB collection interface.

    Abstract interface that ensures compatibility with the ChromaDB
    collection API regardless of implementation details.
    """

    @eidosian()
    def count(self) -> int:
        """Return the number of items in the collection."""
        ...

    @eidosian()
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Insert or update items in the collection."""
        ...

    @eidosian()
    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """Query the collection for similar items."""
        ...

    @eidosian()
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete items from the collection by ID or filter."""
        ...


class ChromaClient(Protocol):
    """
    Protocol defining required ChromaDB client interface.

    Abstract interface that ensures compatibility with the ChromaDB
    client API regardless of implementation details.
    """

    @eidosian()
    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ChromaCollection:
        """Get or create a collection with the given name."""
        ...

    @eidosian()
    def persist(self) -> None:
        """Persist the database to disk."""
        ...


# Using StorageType from centralized config
StorageType = config.vectorizer.storage_type.__class__


class VectorStoreError(DatabaseError):
    """Base exception for vector store operations."""

    pass


class InitializationError(VectorStoreError):
    """
    Raised when vector store initialization fails.

    This occurs when the vector store cannot be properly initialized,
    such as when the embedding model fails to load or the storage
    backend cannot be configured.
    """

    pass


class ModelLoadError(VectorStoreError):
    """
    Raised when embedding model loading fails.

    This occurs when the specified embedding model cannot be loaded,
    such as when the model file is missing or corrupted.
    """

    pass


class UpsertError(VectorStoreError):
    """
    Raised when adding or updating vectors fails.

    This occurs when an attempt to store or update vectors in the
    database fails, such as due to invalid data or storage constraints.
    """

    pass


class SearchError(VectorStoreError):
    """
    Raised when vector similarity search fails.

    This occurs when a semantic search operation cannot be completed,
    such as due to missing or incompatible data.
    """

    pass


class DimensionMismatchError(VectorStoreError):
    """
    Raised when vector dimensions don't match expected dimensions.

    This occurs when the dimension of a provided vector doesn't match
    the expected dimension of the vector store, which would lead to
    incompatible operations.
    """

    pass


class ContentProcessingError(VectorStoreError):
    """
    Raised when processing content for embedding fails.

    This occurs when text content cannot be properly processed into
    vector embeddings, such as due to invalid format or content issues.
    """

    pass


SQLITE_DB_FILENAME = "vector_store.sqlite3"


class SQLiteFAISSCollection:
    """Lightweight SQLite + FAISS collection used when Chroma is unavailable."""

    def __init__(self, db_path: Path, dimension: int) -> None:
        if faiss is None:
            install_hint = 'pip install "word_forge[vector]"'
            missing = _FAISS_IMPORT_ERROR or RuntimeError("faiss unavailable")
            raise InitializationError(
                "SQLite/FAISS persistence requires the faiss-cpu dependency. "
                f"Install it via {install_hint}."
            ) from missing

        self.db_path = db_path
        self.dimension = dimension
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    @eidosian()
    def count(self) -> int:
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM vectors")
            return int(cursor.fetchone()[0])

    @eidosian()
    def persist(self) -> None:
        with self._lock:
            self._conn.commit()

    @eidosian()
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        metadata_list = metadatas or [None] * len(ids)
        documents_list = documents or [None] * len(ids)

        with self._lock:
            for idx, vec_id in enumerate(ids):
                vector = np.asarray(embeddings[idx], dtype=np.float32)
                if vector.shape[0] != self.dimension:
                    raise DimensionMismatchError(
                        f"Vector dimension {vector.shape[0]} does not match {self.dimension}"
                    )

                metadata_blob = (
                    json.dumps(metadata_list[idx]) if metadata_list[idx] else None
                )
                document_value = documents_list[idx]
                payload = sqlite3.Binary(vector.tobytes())

                self._conn.execute(
                    """
                    INSERT INTO vectors(id, embedding, metadata, document)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        embedding=excluded.embedding,
                        metadata=excluded.metadata,
                        document=excluded.document
                    """,
                    (vec_id, payload, metadata_blob, document_value),
                )

            self._conn.commit()

    @eidosian()
    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        if not query_embeddings:
            raise SearchError("SQLite/FAISS backend requires query embeddings")

        query_vector = np.asarray(query_embeddings[0], dtype=np.float32)
        if query_vector.shape[0] != self.dimension:
            raise DimensionMismatchError(
                f"Query dimension {query_vector.shape[0]} does not match {self.dimension}"
            )

        with self._lock:
            rows = self._conn.execute(
                "SELECT id, embedding, metadata, document FROM vectors"
            ).fetchall()

        filtered_rows = self._filter_rows(rows, where)
        if not filtered_rows:
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
            }

        dataset = np.vstack([self._row_to_vector(row) for row in filtered_rows])
        metadatas = [
            self._deserialize_metadata(row["metadata"]) for row in filtered_rows
        ]
        documents = [row["document"] for row in filtered_rows]
        ids = [row["id"] for row in filtered_rows]

        # Normalize for cosine similarity and search with FAISS
        faiss.normalize_L2(dataset)
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(dataset)
        top_k = min(n_results, len(ids))
        distances, neighbors = index.search(query_vector, top_k)

        ordered_ids = [ids[i] for i in neighbors[0]] if top_k else []
        ordered_metadatas = [metadatas[i] for i in neighbors[0]] if top_k else []
        ordered_documents = [documents[i] for i in neighbors[0]] if top_k else []
        ordered_distances = distances[0].tolist() if top_k else []

        return {
            "ids": [ordered_ids],
            "distances": [ordered_distances],
            "metadatas": [ordered_metadatas],
            "documents": [ordered_documents],
        }

    @eidosian()
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if ids:
                self._conn.executemany(
                    "DELETE FROM vectors WHERE id = ?", [(vec_id,) for vec_id in ids]
                )
            elif where:
                rows = self._conn.execute("SELECT id, metadata FROM vectors").fetchall()
                filtered_ids = [row["id"] for row in self._filter_rows(rows, where)]
                if filtered_ids:
                    self._conn.executemany(
                        "DELETE FROM vectors WHERE id = ?",
                        [(vec_id,) for vec_id in filtered_ids],
                    )
            self._conn.commit()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    document TEXT
                )
                """
            )
            self._conn.commit()

    def _row_to_vector(self, row: sqlite3.Row) -> NDArray[np.float32]:
        return np.frombuffer(row["embedding"], dtype=np.float32)

    def _deserialize_metadata(self, payload: Optional[str]) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def _filter_rows(
        self, rows: List[sqlite3.Row], where: Optional[Dict[str, Any]]
    ) -> List[sqlite3.Row]:
        if not where:
            return rows

        filtered: List[sqlite3.Row] = []
        for row in rows:
            metadata = self._deserialize_metadata(row["metadata"]) or {}
            if all(metadata.get(key) == value for key, value in where.items()):
                filtered.append(row)
        return filtered


class InMemoryCollection:
    """Simplified in-memory collection for explicit demo scenarios."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._store: Dict[str, Dict[str, Any]] = {}

    @eidosian()
    def count(self) -> int:
        return len(self._store)

    @eidosian()
    def persist(self) -> None:
        return None

    @eidosian()
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        metadata_list = metadatas or [None] * len(ids)
        documents_list = documents or [None] * len(ids)
        for idx, vec_id in enumerate(ids):
            vector = np.asarray(embeddings[idx], dtype=np.float32)
            if vector.shape[0] != self.dimension:
                raise DimensionMismatchError(
                    f"Vector dimension {vector.shape[0]} does not match {self.dimension}"
                )
            self._store[vec_id] = {
                "embedding": vector,
                "metadata": metadata_list[idx],
                "document": documents_list[idx],
            }

    @eidosian()
    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        if not query_embeddings:
            raise SearchError("Demo collection requires query embeddings")

        query_vector = np.asarray(query_embeddings[0], dtype=np.float32)
        if query_vector.shape[0] != self.dimension:
            raise DimensionMismatchError(
                f"Query dimension {query_vector.shape[0]} does not match {self.dimension}"
            )

        items = list(self._store.items())
        if where:
            items = [
                (vec_id, payload)
                for vec_id, payload in items
                if payload.get("metadata")
                and all(payload["metadata"].get(k) == v for k, v in where.items())
            ]

        if not items:
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
            }

        dataset = np.vstack([payload["embedding"] for _, payload in items])
        top_k = min(n_results, len(items))

        if faiss is not None:
            faiss.normalize_L2(dataset)
            query_norm = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_norm)
            index = faiss.IndexFlatIP(self.dimension)
            index.add(dataset)
            distances, neighbors = index.search(query_norm, top_k)
            ordered_indices = neighbors[0] if top_k else []
            ordered_distances = distances[0].tolist() if top_k else []
        else:  # pragma: no cover - fallback path when faiss missing
            dataset_norm = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
            dataset_norm[np.isnan(dataset_norm)] = 0.0
            query_norm = query_vector / np.linalg.norm(query_vector)
            if np.isnan(query_norm).any():
                query_norm = np.zeros_like(query_norm)
            similarities = dataset_norm @ query_norm.reshape(-1, 1)
            ordered_indices = (
                np.argsort(similarities[:, 0])[::-1][:top_k] if top_k else np.array([])
            )
            ordered_distances = (
                similarities[ordered_indices, 0].tolist() if top_k else []
            )

        ordered = [items[int(i)] for i in ordered_indices] if top_k else []
        ordered_ids = [vec_id for vec_id, _ in ordered]
        ordered_metadatas = [payload.get("metadata") for _, payload in ordered]
        ordered_documents = [payload.get("document") for _, payload in ordered]

        return {
            "ids": [ordered_ids],
            "distances": [ordered_distances],
            "metadatas": [ordered_metadatas],
            "documents": [ordered_documents],
        }

    @eidosian()
    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        if ids:
            for vec_id in ids:
                self._store.pop(vec_id, None)
            return

        if where:
            to_delete = [
                vec_id
                for vec_id, payload in self._store.items()
                if payload.get("metadata")
                and all(payload["metadata"].get(k) == v for k, v in where.items())
            ]
            for vec_id in to_delete:
                self._store.pop(vec_id, None)


# SQL query constants from centralized config
SQL_GET_TERM_BY_ID = config.vectorizer.sql_templates["get_term_by_id"]
SQL_GET_MESSAGE_TEXT = config.vectorizer.sql_templates["get_message_text"]


class VectorStore:
    """
    Universal vector store for all Word Forge linguistic data with multilingual support.

    Provides storage and retrieval for embeddings of diverse linguistic content:
    - Words and their definitions in multiple languages
    - Usage examples
    - Conversation messages
    - Emotional valence and arousal

    Uses the advanced Multilingual-E5-large-instruct model to create contextually rich
    embeddings with instruction-based formatting for optimal retrieval performance.

    Attributes:
        model: SentenceTransformer model for generating embeddings
        dimension: Vector dimension size (typically 1024 for E5 models)
        model_name: Name of the embedding model being used
        client: Backend-specific client for vector storage operations
        collection: Backend collection storing the vectors
        index_path: Path to persistent storage location
        storage_type: Whether using memory or disk storage
        db_manager: Optional database manager for content lookups
        emotion_manager: Optional emotion manager for sentiment analysis
        backend_name: Name of the active storage backend
        demo_mode: Whether the store is running in non-persistent demo mode
    """

    # Declare class attributes with type annotations
    dimension: int
    model: SentenceTransformer
    model_name: str
    client: Union[ChromaClient, SQLiteFAISSCollection, InMemoryCollection]
    collection: Union[ChromaCollection, SQLiteFAISSCollection, InMemoryCollection]
    index_path: Path
    storage_type: StorageType
    db_manager: Optional[DBManager]
    emotion_manager: Optional[EmotionManager]
    logger: logging.Logger
    instruction_templates: Dict[str, InstructionTemplate]
    backend_name: str
    demo_mode: bool

    def __init__(
        self,
        dimension: Optional[int] = None,
        model_name: Optional[str] = None,
        index_path: Optional[Union[str, Path]] = None,
        storage_type: Optional[StorageType] = None,
        collection_name: Optional[str] = None,
        db_manager: Optional[DBManager] = None,
        emotion_manager: Optional[EmotionManager] = None,
        demo_mode: bool = False,
    ):
        """
        Initialize the vector store with specified configuration.

        Sets up the embedding model, storage backend, and all necessary
        components for the vector store to function properly. Connects to
        persistent storage and loads or initializes required models.

        Args:
            dimension: Optional override for vector dimensions. If None, it will be inferred from the model.
            model_name: Optional embedding model to use. Defaults to config.
            index_path: Optional path for vector storage. Defaults to config.
            storage_type: Optional storage type (memory or disk). Defaults to config.
            collection_name: Optional name for the vector collection. Defaults to config or 'word_forge_vectors'.
            db_manager: Optional database manager for content lookup.
            emotion_manager: Optional emotion manager for sentiment analysis.
            demo_mode: Explicit flag to allow ephemeral in-memory demo storage.

        Raises:
            InitializationError: If initialization fails.
            ModelLoadError: If embedding model cannot be loaded or dimension cannot be inferred.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Store configuration, using defaults from config object
        self.index_path = Path(index_path or config.vectorizer.index_path)
        self.storage_type = storage_type or config.vectorizer.storage_type
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager
        self.model_name = model_name or config.vectorizer.model_name
        self._use_ollama = self.model_name.startswith("ollama:")
        self._ollama_model = (
            self.model_name.split("ollama:", 1)[1].strip() if self._use_ollama else ""
        )

        self.demo_mode = bool(demo_mode)
        if self.storage_type == StorageType.MEMORY and not self.demo_mode:
            raise InitializationError(
                "In-memory vector storage is reserved for explicit demo usage. "
                "Set demo_mode=True to acknowledge that vectors will not persist."
            )
        if self.storage_type != StorageType.MEMORY and self.demo_mode:
            self.logger.warning(
                "demo_mode flag was provided but persistent storage is enabled; ignoring demo mode"
            )
            self.demo_mode = False

        # Validate and create storage directory if needed
        if self.storage_type != StorageType.MEMORY:
            os.makedirs(self.index_path, exist_ok=True)

        # Load the embedding model first
        if self._use_ollama:
            if not self._ollama_model:
                raise ModelLoadError("Ollama embedding model name is empty.")
            self.model = None
        else:
            # Ensure core embedding dependency exists
            if SentenceTransformer is None:
                install_hint = 'pip install "word_forge[vector]"'
                raise InitializationError(
                    "VectorStore requires sentence-transformers for embedding support. "
                    f"Install it via {install_hint} to enable semantic search."
                ) from _VECTOR_IMPORT_ERROR
            try:
                # Ensure model is loaded only once if not already present
                if not hasattr(self, "model"):
                    self.model = SentenceTransformer(self.model_name)  # type: ignore
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load embedding model '{self.model_name}': {str(e)}"
                ) from e

        # Determine the vector dimension
        try:
            if dimension is not None:
                self.dimension = dimension
            elif (
                hasattr(config.vectorizer, "dimension")
                and config.vectorizer.dimension is not None
                and config.vectorizer.dimension > 0
            ):
                self.dimension = config.vectorizer.dimension
            else:
                if self._use_ollama:
                    self.dimension = self._infer_ollama_dimension()
                    self.logger.info(
                        "Inferred dimension %s from Ollama model '%s'",
                        self.dimension,
                        self._ollama_model,
                    )
                else:
                    # Infer dimension from the loaded model
                    model_dimension = self.model.get_sentence_embedding_dimension()  # type: ignore
                    if (
                        model_dimension is None
                        or not isinstance(model_dimension, int)
                        or model_dimension <= 0
                    ):
                        raise ModelLoadError(
                            f"Could not infer valid dimension from model '{self.model_name}'"
                        )
                    self.dimension = model_dimension
                    self.logger.info(
                        f"Inferred dimension {self.dimension} from model '{self.model_name}'"
                    )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to determine vector dimension: {str(e)}"
            ) from e

        # Initialize storage backend after dimension is set
        self.backend_name = ""
        backend_errors: List[str] = []

        if self.storage_type == StorageType.MEMORY:
            self.collection = InMemoryCollection(self.dimension)
            self.client = self.collection
            self.backend_name = "memory-demo"
        else:
            chroma_ready = False
            if chromadb is not None:
                try:
                    self.client = self._create_client()
                    collection_name = collection_name or (
                        config.vectorizer.collection_name or "word_forge_vectors"
                    )
                    self.collection = self._initialize_collection(collection_name)
                    self.backend_name = "chromadb"
                    chroma_ready = True
                except Exception as e:
                    backend_errors.append(f"Chroma backend failed: {e}")

            if not chroma_ready:
                try:
                    sqlite_path = self.index_path / SQLITE_DB_FILENAME
                    self.collection = SQLiteFAISSCollection(sqlite_path, self.dimension)
                    self.client = self.collection
                    self.backend_name = "sqlite-faiss"
                except Exception as e:
                    backend_errors.append(f"SQLite/FAISS backend failed: {e}")

            if not self.backend_name:
                combined_error = " | ".join(backend_errors)
                raise InitializationError(
                    "Failed to initialize persistent vector store. " + combined_error
                )

        # Load instruction templates if available
        self.instruction_templates: Dict[str, InstructionTemplate] = {}
        if hasattr(config.vectorizer, "instruction_templates"):
            # Convert the templates to the expected type
            for key, template in config.vectorizer.instruction_templates.items():
                # Cast QueryType key to string
                str_key = str(key)
                # Convert TemplateDict to InstructionTemplate
                self.instruction_templates[str_key] = cast(
                    InstructionTemplate, template
                )

        self.logger.info(
            f"VectorStore initialized: model={self.model_name}, "
            f"dimension={self.dimension}, storage={self.storage_type.name.lower()}, "
            f"backend={self.backend_name}, path={self.index_path}, demo_mode={self.demo_mode}"
        )

    def _create_client(self) -> ChromaClient:
        """
        Create an appropriate ChromaDB client based on configuration.

        Returns:
            ChromaClient: Configured ChromaDB client for vector operations

        Raises:
            InitializationError: If client creation fails
        """
        try:
            if self.storage_type == StorageType.MEMORY:
                return cast(ChromaClient, chromadb.Client())
            else:
                return cast(
                    ChromaClient, chromadb.PersistentClient(path=str(self.index_path))
                )
        except Exception as e:
            raise InitializationError(
                f"Failed to create ChromaDB client: {str(e)}"
            ) from e

    def _initialize_collection(self, collection_name: str) -> ChromaCollection:
        """
        Initialize or connect to a ChromaDB collection.

        Args:
            collection_name: Name of the collection to initialize

        Returns:
            ChromaCollection: ChromaDB collection for vector operations

        Raises:
            InitializationError: If collection initialization fails
        """
        try:
            # Create or get collection
            metadata: Dict[str, Any] = {
                "dimension": self.dimension,
                "model": self.model_name,
            }
            collection = self.client.get_or_create_collection(
                collection_name, metadata=metadata
            )
            self.logger.info(
                f"Connected to collection '{collection_name}' with {collection.count()} vectors"
            )
            return collection
        except Exception as e:
            raise InitializationError(
                f"Failed to initialize collection: {str(e)}"
            ) from e

    def _validate_vector_dimension(
        self, vector: NDArray[np.float32], context: str = "Vector"
    ) -> None:
        """
        Validate that vector dimensions match expected dimensions.

        Args:
            vector: The vector to validate
            context: Description of the vector for error messages

        Raises:
            DimensionMismatchError: If vector dimensions are incorrect
        """
        length = (
            len(vector)
            if not hasattr(vector, "shape")
            else (vector.shape[0] if len(getattr(vector, "shape")) > 0 else 0)
        )
        if length != self.dimension:
            raise DimensionMismatchError(
                f"{context} dimension {length} doesn't match expected {self.dimension}"
            )

    def _normalize_vector_dimension(
        self, vector: NDArray[np.float32], context: str
    ) -> NDArray[np.float32]:
        """
        Coerce an embedding to the configured dimension by padding or truncating.

        When upstream models change or cached indices expect a different shape,
        resize the vector and re-normalize it to avoid runtime failures while
        keeping cosine similarity meaningful.

        Args:
            vector: Incoming embedding
            context: Human-friendly source label for logging

        Returns:
            NDArray[np.float32]: Dimensionally aligned and normalized vector
        """
        # Handle both numpy arrays and plain lists/sequences
        if hasattr(vector, "shape"):
            length = vector.shape[0] if len(vector.shape) > 0 else 0
        else:
            length = len(vector)
            # Convert to numpy array if not already
            vector = np.asarray(vector, dtype=np.float32)

        if length == self.dimension:
            return vector

        if length == 0:
            raise DimensionMismatchError(
                f"{context} is empty; expected {self.dimension}"
            )

        if length > self.dimension:
            coerced = vector[: self.dimension]
            self.logger.debug(
                "%s truncated from dimension %s to %s", context, length, self.dimension
            )
        else:
            pad_width = self.dimension - length
            coerced = np.pad(vector, (0, pad_width), mode="constant")
            self.logger.debug(
                "%s padded from dimension %s to %s with zeros",
                context,
                length,
                self.dimension,
            )

        # Renormalize to keep cosine similarity stable
        norm = np.linalg.norm(coerced)
        if norm > 0:
            coerced = coerced / norm

        return coerced.astype(np.float32)

    @eidosian()
    def format_with_instruction(
        self, text: str, template_key: str = "search", is_query: bool = True
    ) -> str:
        """
        Format text with instruction templates for embedding models.

        Applies the appropriate instruction template to optimize the text
        for the specific embedding model being used. This is especially
        important for instruction-tuned models that expect specific formats.

        Args:
            text: Text to format
            template_key: Template type to use (search, definition, etc.)
            is_query: Whether this is a query (vs. document)

        Returns:
            Formatted text ready for embedding
        """
        # Don't format if no templates available
        if not self.instruction_templates:
            return text
        # Get template or use default
        # Get template or use default
        template: Optional[InstructionTemplate] = self.instruction_templates.get(
            template_key, self.instruction_templates.get("default", None)
        )

        # Return unformatted text if template is None
        if template is None:
            return text

        # Format using appropriate template parts
        task = template.get("task", "")
        prefix = template.get("query_prefix" if is_query else "document_prefix", "")

        if task and prefix:
            return f"{task}\n{prefix}{text}"
        elif prefix:
            return f"{prefix}{text}"
        else:
            return text

    @eidosian()
    def embed_text(
        self,
        text: str,
        template_key: str = "search",
        is_query: bool = True,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """
        Transforms raw text into a high-dimensional vector representation

        Applies the appropriate instruction template based on the task type,
        then generates a vector embedding using the configured model.

        Args:
            text: Text to embed
            template_key: Template type to use (search, definition, etc.)
            is_query: Whether this is a query (True) or document (False)
            normalize: Whether to normalize the vector to unit length

        Returns:
            Embedding vector as numpy array (normalized by default)

        Raises:
            ContentProcessingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ContentProcessingError("Cannot embed empty text")

        try:
            # Format with instruction template if available
            formatted_text = self.format_with_instruction(text, template_key, is_query)

            # Generate embedding
            if self._use_ollama:
                vector = self._embed_text_ollama(formatted_text)
                if normalize:
                    vector = self._normalize_vector_dimension(vector, context="Ollama embedding")
            else:
                # Cast is appropriate here as SentenceTransformer.encode can return different types
                vector = cast(
                    NDArray[np.float32],
                    self.model.encode(  # type: ignore
                        formatted_text,
                        normalize_embeddings=normalize,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ),
                )

            # Ensure correct type
            vector = vector.astype(np.float32)

            # Normalize and validate dimensions
            vector = self._normalize_vector_dimension(
                vector,
                context=f"Embedding for '{text[:30]}{'...' if len(text) > 30 else ''}'",
            )
            self._validate_vector_dimension(
                vector,
                context=f"Embedding for '{text[:30]}{'...' if len(text) > 30 else ''}'",
            )

            return vector

        except Exception as e:
            raise ContentProcessingError(f"Failed to embed text: {str(e)}") from e

    def _embed_text_ollama(self, text: str) -> NDArray[np.float32]:
        """Generate embeddings via local Ollama server."""
        try:
            import requests  # type: ignore

            payload = {"model": self._ollama_model, "prompt": text}
            resp = requests.post(
                "http://localhost:11434/api/embeddings",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise ContentProcessingError("Ollama returned invalid embedding payload.")
            vector = np.asarray(embedding, dtype=np.float32)
            return vector
        except Exception as e:
            raise ContentProcessingError(f"Ollama embedding failed: {str(e)}") from e

    def _infer_ollama_dimension(self) -> int:
        """Probe Ollama embedding dimension with a tiny request."""
        probe = self._embed_text_ollama("dimension probe")
        if probe.ndim != 1 or probe.shape[0] <= 0:
            raise ModelLoadError(
                f"Could not infer valid dimension from Ollama model '{self._ollama_model}'"
            )
        return int(probe.shape[0])

    def _get_content_info(
        self, content_id: int, content_type: ContentType
    ) -> Dict[str, Any]:
        """
        Retrieve additional information about content for metadata.

        Args:
            content_id: ID of the content item
            content_type: Type of content (word, definition, etc.)

        Returns:
            Dict containing metadata about the content

        Raises:
            ContentProcessingError: If content retrieval fails
        """
        # Skip if no database manager or no connection
        if self.db_manager is None or self.db_manager.connection is None:
            return {}

        try:
            if content_type == "word":
                # Get the word term from the ID
                cursor = self.db_manager.connection.execute(
                    SQL_GET_TERM_BY_ID, (content_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {"term": row[0]}

            elif content_type == "message":
                # Get message text and other metadata
                cursor = self.db_manager.connection.execute(
                    SQL_GET_MESSAGE_TEXT, (content_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "text": row[0],
                        "conversation_id": row[1] if len(row) > 1 else None,
                        "speaker": row[2] if len(row) > 2 else None,
                    }

            return {}

        except Exception as e:
            self.logger.warning(
                f"Failed to get content info for {content_type} ID {content_id}: {e}"
            )
            return {}

    def _get_emotion_info(self, item_id: int) -> Dict[str, Any]:
        """
        Get emotional attributes for an item if available.

        Args:
            item_id: ID of the item to get emotions for

        Returns:
            Dict containing emotion data (valence, arousal, etc.)
        """
        # Skip if no emotion manager
        if self.emotion_manager is None:
            return {}

        try:
            # Get emotion data from the emotion manager
            emotion_data = self.emotion_manager.get_word_emotion(item_id)
            if emotion_data:
                return {
                    "emotion_valence": emotion_data.get("valence"),
                    "emotion_arousal": emotion_data.get("arousal"),
                    "emotion_label": emotion_data.get("label"),
                }
            return {}

        except Exception as e:
            self.logger.warning(f"Failed to get emotion info for ID {item_id}: {e}")
            return {}

    @eidosian()
    def process_word_entry(self, entry: WordEntryDict) -> List[VectorItem]:
        """
        Process a word entry into vector items for storage.

        Creates separate vector items for the word, its definition, and usage examples,
        each with appropriate metadata for filtering and retrieval.

        Args:
            entry: Word entry dictionary with all word data

        Returns:
            List of items ready for vector storage, each containing vector, metadata, and text

        Raises:
            ContentProcessingError: If processing fails
        """
        word_id = entry["id"]
        term = entry["term"]
        definition = entry["definition"]

        # Handle usage_examples that could be either string or list
        usage_examples = entry.get("usage_examples", "")
        usage_examples = "; ".join(usage_examples)
        examples = self._parse_usage_examples(usage_examples)

        language = entry.get("language", "en")

        # Items to be vectorized
        vector_items: List[VectorItem] = []

        try:
            # 1. Process the word term itself
            word_embedding = self.embed_text(term, template_key="term", normalize=True)

            # Combine standard metadata with emotion data
            emotion_info = self._get_emotion_info(int(word_id))
            word_metadata: VectorMetadata = {
                "original_id": int(word_id),
                "content_type": "word",
                "term": term,
                "language": language,
                "timestamp": time.time(),
                "emotion_valence": emotion_info.get("emotion_valence"),
                "emotion_arousal": emotion_info.get("emotion_arousal"),
                "emotion_label": emotion_info.get("emotion_label"),
            }

            # Add word item
            vector_items.append(
                {
                    "id": f"w_{word_id}",
                    "text": term,
                    "metadata": word_metadata,
                    "vector": word_embedding,
                }
            )

            # 2. Process the definition if available
            if definition and definition.strip():
                def_embedding = self.embed_text(
                    definition,
                    template_key="definition",
                    normalize=True,
                    is_query=False,
                )

                def_metadata: VectorMetadata = {
                    "original_id": int(word_id),
                    "content_type": "definition",
                    "term": term,
                    "definition": definition,
                    "language": language,
                    "timestamp": time.time(),
                }

                # Add definition item
                vector_items.append(
                    {
                        "id": f"d_{word_id}",
                        "text": definition,
                        "metadata": def_metadata,
                        "vector": def_embedding,
                    }
                )

            # 3. Process each usage example
            for i, example in enumerate(examples):
                if not example.strip():
                    continue

                example_embedding = self.embed_text(
                    example, template_key="example", normalize=True, is_query=False
                )

                example_metadata: VectorMetadata = {
                    "original_id": int(word_id),
                    "content_type": "example",
                    "term": term,
                    "language": language,
                    "timestamp": time.time(),
                }

                # Add example item
                vector_items.append(
                    {
                        "id": f"e_{word_id}_{i}",
                        "text": example,
                        "metadata": example_metadata,
                        "vector": example_embedding,
                    }
                )

            return vector_items

        except Exception as e:
            raise ContentProcessingError(
                f"Failed to process word entry for '{term}': {str(e)}"
            ) from e

    def _parse_usage_examples(self, examples_string: str) -> List[str]:
        """
        Parse usage examples from a semicolon-separated string.

        Args:
            examples_string: String containing semicolon-separated examples

        Returns:
            List of individual usage examples
        """
        if not examples_string:
            return []

        # Split by semicolon and strip whitespace
        examples = [ex.strip() for ex in examples_string.split(";")]

        # Filter out empty examples
        return [ex for ex in examples if ex]

    @eidosian()
    def store_word(self, entry: WordEntryDict) -> int:
        """
        Creates vector embeddings for the word term, definition, and usage examples,
        then stores them in the vector database with appropriate metadata.

        This is the main entry point for adding word data to the vector store.

        Args:
            entry: Complete word entry with all data

        Returns:
            int: Number of vectors successfully stored

        Raises:
            ContentProcessingError: If processing or storage fails
        """
        try:
            # Process the word entry into vector items
            vector_items = self.process_word_entry(entry)

            # Store each vector item
            for item in vector_items:
                # Extract vector_id from the id string (e.g., "w_123" -> 123)
                vec_id = item["id"]

                # Convert VectorMetadata to Dict[str, Any]
                metadata_dict = dict(item["metadata"])

                # Call upsert with extracted ID and other components
                self.upsert(
                    vec_id=vec_id,
                    embedding=item["vector"],
                    metadata=metadata_dict,
                    text=item["text"],
                )

            # Return the number of vectors stored
            return len(vector_items)

        except Exception as e:
            raise ContentProcessingError(
                f"Failed to store word '{entry.get('term', '')}': {str(e)}"
            ) from e

    @eidosian()
    def delete_vectors_for_word(self, word_id: WordID) -> int:
        """
        Delete all vectors associated with a specific word.

        Removes vectors associated with the word from the vector store,
        including all vectors for the term, definition, and examples.

        Args:
            word_id: ID of the word whose vectors should be deleted

        Returns:
            Number of vectors deleted

        Raises:
            VectorStoreError: If the deletion operation fails

        Examples:
            >>> store = VectorStore(dimension=384)
            >>> # After adding vectors for a word
            >>> deleted_count = store.delete_vectors_for_word(123)
            >>> print(f"Deleted {deleted_count} vectors")
        """
        try:
            # Convert to string for compatibility with ChromaDB
            word_id_str = str(word_id)
            deleted_count = 0

            # Try to delete vectors by ID pattern
            try:
                # Use metadata filter to find all vectors for this word
                word_filter = {"original_id": int(word_id)}

                # Try to delete vectors by ID and filter
                self.collection.delete(where=word_filter)

                # Count the number of vectors deleted
                deleted_count = 3  # Approximate for term, definition, examples
                self.logger.info(f"Deleted vectors for word {word_id} using filter")

            except Exception as inner_e:
                self.logger.warning(
                    f"Failed to delete vectors by filter for {word_id}: {inner_e}"
                )

                # Fallback: Try explicit ID patterns
                ids_to_delete = [
                    f"w_{word_id_str}",  # Word term
                    f"d_{word_id_str}",  # Definition
                ]

                # Add potential example IDs
                for i in range(10):  # Assume max 10 examples per word
                    ids_to_delete.append(f"e_{word_id_str}_{i}")

                try:
                    self.collection.delete(ids=ids_to_delete)
                    deleted_count = len(ids_to_delete)
                    self.logger.info(
                        f"Deleted vectors for word {word_id} using explicit IDs"
                    )
                except Exception as id_e:
                    self.logger.error(
                        f"Failed to delete vectors by ID for {word_id}: {id_e}"
                    )
                    raise VectorStoreError(
                        "Neither ID nor filter-based deletion succeeded"
                    )

            return deleted_count

        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete vectors for word {word_id}: {str(e)}"
            ) from e

    @eidosian()
    def upsert(
        self,
        vec_id: VectorID,
        embedding: NDArray[np.float32],
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> None:
        """
        Add or update a vector in the store with metadata and optional text.

        This function validates the vector dimension, sanitizes metadata to ensure
        ChromaDB compatibility, and handles the low-level storage operations.

        Args:
            vec_id: Unique identifier for the vector
            embedding: Vector embedding to store
            metadata: Optional metadata to associate with the vector
            text: Optional text content to associate with the vector

        Raises:
            UpsertError: If ChromaDB operation fails
        """
        # Normalize and validate vector dimensions
        embedding = self._normalize_vector_dimension(embedding, "Embedding")
        self._validate_vector_dimension(embedding, "Embedding")

        # Convert ID to string for ChromaDB
        vec_id_str = str(vec_id)

        # Prepare metadata - ensure all values are compatible with ChromaDB
        sanitized_metadata = self._sanitize_metadata(metadata or {})

        try:
            if isinstance(self.collection, dict):
                self.collection[vec_id_str] = {
                    "embedding": embedding,
                    "metadata": sanitized_metadata,
                    "text": text,
                }
            else:
                self.collection.upsert(
                    ids=[vec_id_str],
                    embeddings=[embedding.tolist()],
                    metadatas=[sanitized_metadata] if sanitized_metadata else None,
                    documents=[text] if text else None,
                )

                # Persist to disk if using persistent storage
                self._persist_if_needed()

        except Exception as e:
            raise UpsertError(f"Failed to store vector: {str(e)}") from e

    def _sanitize_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Sanitize metadata to ensure compatibility with ChromaDB.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Dictionary with ChromaDB-compatible values
        """
        result: Dict[str, Union[str, int, float, bool]] = {}

        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Convert values to compatible types
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                # Convert other types to string
                result[key] = str(value)

        return result

    # Type overloads for search method to enable different call patterns
    @overload
    def search(
        self,
        *,
        query_text: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResultDict]: ...

    @overload
    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResultDict]: ...

    @eidosian()
    def search(
        self,
        query_vector: Optional[NDArray[np.float32]] = None,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        *,
        query_text: Optional[str] = None,
    ) -> List[SearchResultDict]:
        """
        Search for semantically similar vectors using vector or text query.

        Performs a similarity search against the vector database, finding items
        that are semantically similar to the query. Supports two search modes:

        1. Direct vector search - when you already have an embedding
        2. Text search - converts text to embedding then searches

        Args:
            query_vector: Vector representation to search against
            query_text: Text to search for (will be converted to a vector)
            k: Number of results to return
            filter_metadata: Optional metadata filters for narrowing results

        Returns:
            List of search results with distance scores and metadata

        Raises:
            SearchError: If neither query_vector nor query_text is provided
            DimensionMismatchError: If query vector dimensions are incorrect

        Examples:
            >>> # Search by text
            >>> results = vector_store.search(
            ...     query_text="machine learning algorithm",
            ...     k=5,
            ...     filter_metadata={"content_type": "definition"}
            ... )

            >>> # Search by vector
            >>> results = vector_store.search(query_vector=embedding)
        """
        if query_vector is None and query_text is None:
            raise SearchError(
                "Search requires either query_vector or query_text parameter"
            )

        if query_vector is not None and query_text is not None:
            raise SearchError("Provide either query_vector or query_text, not both")

        # Prepare search vector
        search_vector = self._prepare_search_vector(query_vector, query_text)

        # Execute the search
        return self._execute_vector_search(
            search_vector=search_vector, k=k, filter_metadata=filter_metadata
        )

    def _prepare_search_vector(
        self, query_vector: Optional[NDArray[np.float32]], query_text: Optional[str]
    ) -> NDArray[np.float32]:
        """
        Prepare the search vector from either direct vector or text input.

        Args:
            query_vector: Pre-computed vector embedding if available
            query_text: Text to embed if vector not provided

        Returns:
            NDArray[np.float32]: Vector to use for similarity search

        Raises:
            SearchError: If neither input is provided
            DimensionMismatchError: If vector has incorrect dimensions
        """
        # Case 1: Direct vector provided
        if query_vector is not None:
            normalized = self._normalize_vector_dimension(
                query_vector, context="Query vector"
            )
            self._validate_vector_dimension(normalized, context="Query vector")
            return normalized

        # Case 2: Text provided - convert to vector
        assert query_text is not None, "Both query_vector and query_text cannot be None"

        try:
            embedded_vector = self.embed_text(
                query_text, template_key="search", is_query=True, normalize=True
            )
            normalized = self._normalize_vector_dimension(
                embedded_vector, context="Embedded query"
            )
            self._validate_vector_dimension(normalized, context="Embedded query")
            return normalized

        except Exception as e:
            raise SearchError(f"Failed to prepare search vector: {str(e)}") from e

    def _execute_vector_search(
        self,
        search_vector: NDArray[np.float32],
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[SearchResultDict]:
        """
        Execute similarity search against the vector database.

        Args:
            search_vector: Vector to search against
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results with distance scores and metadata

        Raises:
            SearchError: If ChromaDB operation fails
        """
        try:
            # Enforce reasonable upper limit on results
            max_allowed = getattr(config.vectorizer, "max_results", 100)
            adjusted_k = min(k, max_allowed)

            # Adjust filter format for ChromaDB
            where_clause: Optional[Dict[str, Any]] = None
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    if value is not None:
                        where_clause[key] = value

            # Execute search (measure time for logging)
            start_time = time.time()

            query_results = self.collection.query(
                query_embeddings=[search_vector.tolist()],
                n_results=adjusted_k,
                where=where_clause,
            )

            search_time = time.time() - start_time

            # Process results into a consistent format
            results = self._process_chromadb_results(query_results)

            # Log search statistics
            self.logger.debug(
                f"Vector search took {search_time:.2f}s for k={adjusted_k}, "
                f"found {len(results)} results"
            )

            return results

        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            self.logger.error(
                f"{error_msg} | Vector shape: {search_vector.shape} | "
                f"Filter: {filter_metadata}"
            )
            raise SearchError(error_msg) from e

    def _process_chromadb_results(
        self, query_results: Dict[str, List[Any]]
    ) -> List[SearchResultDict]:
        """
        Process raw ChromaDB query results into a standardized format.

        Args:
            query_results: Raw results from ChromaDB query

        Returns:
            List of standardized search result dictionaries
        """
        results: List[SearchResultDict] = []

        # Process results if available
        if not query_results or "ids" not in query_results:
            return results

        # Extract result components
        # Cast is appropriate here due to ChromaDB's loose return types
        ids = cast(List[str], query_results.get("ids", [[]])[0])
        distances = cast(List[float], query_results.get("distances", [[]])[0])
        metadatas = cast(
            List[Optional[Dict[str, Any]]],
            query_results.get("metadatas", [[None] * len(ids)])[0],
        )
        documents = cast(
            List[Optional[str]], query_results.get("documents", [[None] * len(ids)])[0]
        )

        # Convert similarities to distances if needed
        if distances and min(distances) >= 0 and max(distances) <= 1:
            distances = self._convert_similarities_to_distances(distances)

        # Build result list
        for i, result_id in enumerate(ids):
            try:
                # Extract the original numeric ID if possible
                original_id = (
                    int(result_id.split("_")[1]) if "_" in result_id else int(result_id)
                )

                # Get metadata and text
                # Cast metadata to the expected type
                metadata = (
                    cast(Optional[VectorMetadata], metadatas[i])
                    if i < len(metadatas)
                    else None
                )
                document = documents[i] if i < len(documents) else None

                # Calculate distance (handle any missing values)
                distance = distances[i] if i < len(distances) else 1.0

                # Add to results
                results.append(
                    SearchResultDict(
                        id=original_id,
                        distance=distance,
                        metadata=metadata,
                        text=document,
                    )
                )
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error processing search result {result_id}: {e}")

        return results

    @eidosian()
    def get_legacy_search_results(
        self, query_vector: NDArray[np.float32], k: int = 5
    ) -> SearchResultList:  # Use renamed TypeAlias
        """
        Legacy method for backward compatibility with older code.

        Returns search results in the original (id, distance) tuple format.

        Args:
            query_vector: Query embedding to search against
            k: Number of results to return

        Returns:
            List of (id, distance) tuples

        Raises:
            SearchError: If search fails
        """
        results = self.search(query_vector=query_vector, k=k)

        # Convert to legacy format
        legacy_results: SearchResultList = [  # Use renamed TypeAlias
            (result["id"], result["distance"]) for result in results
        ]

        return legacy_results

    def _convert_similarities_to_distances(
        self, similarities: List[float]
    ) -> List[float]:
        """
        Convert similarity scores (0-1, higher is better) to distances (lower is better).

        Args:
            similarities: List of similarity scores

        Returns:
            List of equivalent distance scores
        """
        return [1.0 - sim for sim in similarities]

    def _persist_if_needed(self) -> None:
        """
        Persist vector store to disk when appropriate.

        Only applies to persistent storage types.
        """
        if self.storage_type != StorageType.MEMORY and hasattr(self.client, "persist"):
            try:
                self.client.persist()
            except Exception as e:
                self.logger.warning(
                    f"Failed to persist vector store to {self.index_path}: {str(e)}"
                )

    def __del__(self) -> None:
        """
        Ensures vector data is persisted before the object is destroyed

        Called automatically when the object is garbage collected.
        """
        try:
            self._persist_if_needed()
        except Exception:
            # Ignore errors during cleanup
            pass

    @staticmethod
    def is_valid_vector_store(obj: Any) -> bool:
        """
        Check if an object is a valid VectorStore instance.

        Args:
            obj: Object to check

        Returns:
            bool: True if the object is a valid VectorStore, False otherwise
        """
        if not isinstance(obj, VectorStore):
            return False

        # Additional checks for required attributes and methods
        required_attrs = ["upsert", "search", "delete_vectors_for_word"]
        return all(hasattr(obj, attr) for attr in required_attrs)
