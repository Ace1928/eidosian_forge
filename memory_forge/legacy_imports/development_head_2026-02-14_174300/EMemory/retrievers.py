#!/usr/bin/env python3
"""
🔹 Advanced Eidosian Retrievers Module 🔹

This module provides two main classes for document retrieval:

1) SimpleEmbeddingRetriever:
   • Maintains an in-memory list of documents and their embeddings.
   • Uses SentenceTransformer embeddings + cosine similarity to rank relevance.
   • Features batch processing, caching, and flexible similarity metrics.

2) ChromaRetriever:
   • Stores documents in a ChromaDB vector collection for scalable vector searches.
   • Supports metadata storage, where lists are serialized into comma-separated strings.
   • Now with robust error handling and advanced filtering capabilities.

Both classes can be integrated into a hybrid search pipeline, or used independently.
They reflect best practices for robust, production-ready, and Eidosian code:
   • Modular & Extensible
   • Thoroughly Documented
   • Proper Error Handling
   • Clear & Consistent Code Style
"""

import datetime
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Import with fallbacks to handle import errors gracefully
try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class RetrieverError(Exception):
    """Base exception for all retriever-related errors."""

    pass


class EmbeddingError(RetrieverError):
    """Error during embedding generation."""

    pass


class StorageError(RetrieverError):
    """Error accessing or storing documents."""

    pass


class QueryError(RetrieverError):
    """Error during query execution."""

    pass


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizes the given text into a list of tokens/words using NLTK.

    Args:
        text (str): The text to tokenize.

    Returns:
        List[str]: The list of tokenized words.
    """
    if NLTK_AVAILABLE:
        return word_tokenize(text)
    else:
        # Fallback to basic tokenization if NLTK is not available
        return text.lower().split()


def remove_stopwords(tokens: List[str], language: str = "english") -> List[str]:
    """
    Remove common stopwords from a list of tokens.

    Args:
        tokens: List of tokens to filter
        language: Language for stopwords (defaults to 'english')

    Returns:
        List of tokens with stopwords removed
    """
    if NLTK_AVAILABLE:
        try:
            stops = set(stopwords.words(language))
            return [token for token in tokens if token.lower() not in stops]
        except Exception as e:
            logger.warning(f"Error removing stopwords: {e}")
            return tokens
    return tokens


def calculate_embedding_memory(num_docs: int, embedding_dim: int = 384) -> str:
    """
    Calculate approximate memory usage for embeddings.

    Args:
        num_docs: Number of documents
        embedding_dim: Dimensionality of the embeddings (default: 384 for MiniLM)

    Returns:
        String with memory usage estimate
    """
    # Each float takes 4 bytes
    bytes_per_vector = embedding_dim * 4
    total_bytes = num_docs * bytes_per_vector

    # Convert to appropriate unit
    if total_bytes < 1024:
        return f"{total_bytes} bytes"
    elif total_bytes < 1024**2:
        return f"{total_bytes/1024:.2f} KB"
    elif total_bytes < 1024**3:
        return f"{total_bytes/(1024**2):.2f} MB"
    else:
        return f"{total_bytes/(1024**3):.2f} GB"


@dataclass
class PerformanceMetrics:
    """Stores performance metrics for retriever operations."""

    embedding_time: float = 0.0
    search_time: float = 0.0
    doc_count: int = 0
    total_searches: int = 0

    def log_embedding(self, duration: float, doc_count: int = 1) -> None:
        """Record embedding generation metrics."""
        self.embedding_time += duration
        self.doc_count += doc_count

    def log_search(self, duration: float) -> None:
        """Record search operation metrics."""
        self.search_time += duration
        self.total_searches += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_embedding_time = self.embedding_time / max(1, self.doc_count)
        avg_search_time = self.search_time / max(1, self.total_searches)

        return {
            "doc_count": self.doc_count,
            "total_searches": self.total_searches,
            "total_embedding_time": self.embedding_time,
            "total_search_time": self.search_time,
            "avg_embedding_time": avg_embedding_time,
            "avg_search_time": avg_search_time,
            "estimated_memory": calculate_embedding_memory(self.doc_count),
        }


# -----------------------------------------------------------------------------
# Class: SimpleEmbeddingRetriever
# -----------------------------------------------------------------------------
class SimpleEmbeddingRetriever:
    """
    Maintains an in-memory list of documents along with their embeddings,
    providing a straightforward cosine-similarity-based retrieval.

    Usage:
        retriever = SimpleEmbeddingRetriever(model_name='all-MiniLM-L6-v2')
        retriever.add_document("This is a sample document.")
        results = retriever.search("sample query", top_k=5)

    Enhanced features:
        - Document batch processing
        - Document IDs tracking
        - Multiple similarity metrics
        - Performance metrics
        - Embedding caching
        - Text preprocessing options
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None,
        similarity_metric: str = "cosine",
    ) -> None:
        """
        Initializes the retriever by loading a SentenceTransformer model
        for embedding generation.

        Args:
            model_name: Name/path of a pre-trained SentenceTransformer model.
                        Defaults to "all-MiniLM-L6-v2".
            normalize_embeddings: Whether to normalize embeddings for better similarity.
            cache_dir: Directory for caching embeddings (None for no caching).
            similarity_metric: Metric for similarity calculation ('cosine' or 'euclidean').
        """
        self.model_name = model_name
        self.normalize = normalize_embeddings
        self.similarity_metric = similarity_metric
        self.cache_dir = cache_dir
        self._setup_cache_dir()

        # Initialize model with exception handling
        try:
            self.model = SentenceTransformer(model_name)
            if normalize_embeddings:
                logger.info(f"Embeddings will be normalized for {model_name}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise EmbeddingError(f"Error loading embedding model: {e}")

        # Data storage
        self.documents: List[str] = []
        self.doc_ids: List[str] = []  # Track document IDs (optional)
        self.embeddings: Optional[np.ndarray] = None
        self.metrics = PerformanceMetrics()

        # Cache for recently used embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        logger.debug(f"SimpleEmbeddingRetriever initialized with model: {model_name}")

    def _setup_cache_dir(self) -> None:
        """Set up embedding cache directory if specified."""
        if self.cache_dir:
            cache_path = Path(self.cache_dir)
            try:
                cache_path.mkdir(exist_ok=True, parents=True)
                logger.debug(f"Embedding cache directory set up at {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Could not set up cache directory: {e}")
                self.cache_dir = None

    def _compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for a text, with caching if enabled.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first if available
        if self.cache_dir:
            cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")

            # Try loading from file cache
            try:
                if os.path.exists(cache_file):
                    embedding = np.load(cache_file)
                    self._cache_hits += 1
                    logger.debug(f"Loading embedding from cache: {cache_key}")
                    return embedding
            except Exception as e:
                logger.warning(f"Error loading cached embedding: {e}")

        # Check memory cache
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            logger.debug("Cache hit for document embedding")
            return self._embedding_cache[cache_key]

        # Compute new embedding
        self._cache_misses += 1
        try:
            start_time = time.time()
            embedding = self.model.encode([text], normalize_embeddings=self.normalize)[0]

            # Record metrics
            duration = time.time() - start_time
            self.metrics.log_embedding(duration)

            # Store in memory cache (limited size)
            if len(self._embedding_cache) < 1000:  # Limit cache size
                self._embedding_cache[cache_key] = embedding

            # Store in file cache if enabled
            if self.cache_dir:
                try:
                    np.save(cache_file, embedding)
                except Exception as e:
                    logger.warning(f"Failed to cache embedding: {e}")

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    def add_document(self, document: str, doc_id: Optional[str] = None) -> Optional[str]:
        """
        Adds a new document to the in-memory store and updates the embedding matrix.

        Args:
            document: The text content of the document to store.
            doc_id: Optional unique identifier for the document.

        Returns:
            The document ID (generated if not provided).
        """
        if not document:
            logger.warning("Attempted to add empty document, skipping")
            return None

        # Generate ID if not provided
        if doc_id is None:
            doc_id = f"doc_{len(self.documents)}"

        try:
            # Compute embedding for the new document
            new_embedding = self._compute_embedding(document)

            # Add document and its ID
            self.documents.append(document)
            self.doc_ids.append(doc_id)

            # Update embeddings matrix
            if self.embeddings is None:
                # First document
                self.embeddings = np.expand_dims(new_embedding, axis=0)
            else:
                # Add to existing matrix
                self.embeddings = np.vstack([self.embeddings, new_embedding])

            logger.debug(f"Document added with ID {doc_id}. Total documents: {len(self.documents)}")
            return doc_id

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return None

    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add multiple documents in batch for efficiency.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs (generated if not provided)

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("Empty documents list provided to add_documents")
            return []

        # Generate IDs if not provided
        if doc_ids is None:
            start_idx = len(self.documents)
            doc_ids = [f"doc_{i}" for i in range(start_idx, start_idx + len(documents))]
        elif len(doc_ids) != len(documents):
            logger.warning("Length mismatch between documents and doc_ids")
            # Generate missing IDs
            start_idx = len(self.documents)
            doc_ids = doc_ids + [f"doc_{i}" for i in range(start_idx + len(doc_ids), start_idx + len(documents))]

        try:
            # Process batch of documents
            start_time = time.time()
            new_embeddings = self.model.encode(
                documents, normalize_embeddings=self.normalize, show_progress_bar=len(documents) > 10
            )

            # Record metrics
            duration = time.time() - start_time
            self.metrics.log_embedding(duration, len(documents))

            # Update storage
            self.documents.extend(documents)
            self.doc_ids.extend(doc_ids)

            # Update embeddings matrix
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

            logger.debug(f"Added {len(documents)} documents in batch. Total: {len(self.documents)}")
            return doc_ids

        except Exception as e:
            logger.error(f"Error in batch document processing: {e}")
            raise StorageError(f"Failed to add documents in batch: {e}")

    def _calculate_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate similarity between query embedding and document embeddings.

        Args:
            query_embedding: Query embedding vector

        Returns:
            Array of similarity scores
        """
        if self.similarity_metric == "euclidean":
            # Lower distance = higher similarity, so we negate
            distances = euclidean_distances([query_embedding], self.embeddings)[0]
            # Convert to similarity (higher is better)
            similarities = 1 / (1 + distances)
        else:
            # Default: cosine similarity
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        return similarities

    def search(self, query: str, top_k: int = 5, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Searches for documents similar to the given query using the selected
        similarity metric.

        Args:
            query: The search query.
            top_k: Number of top results to return (default is 5).
            threshold: Optional similarity threshold (0-1) to filter results.

        Returns:
            List of dicts, each containing "content", "score", and "id" if available.
        """
        if not self.documents or self.embeddings is None:
            logger.info("No documents stored. Cannot perform search.")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []

        try:
            # Time the search operation
            start_time = time.time()

            # Encode query
            query_embedding = self._compute_embedding(query)

            # Calculate similarity
            similarities = self._calculate_similarity(query_embedding)

            # Filter by threshold if provided
            if threshold is not None:
                mask = similarities >= threshold
                filtered_indices = np.where(mask)[0]

                # Sort filtered indices by similarity
                sorted_indices = filtered_indices[np.argsort(-similarities[filtered_indices])]
                top_indices = sorted_indices[:top_k]
            else:
                # Get top k results by descending similarity
                top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Record search metrics
            duration = time.time() - start_time
            self.metrics.log_search(duration)

            # Format results
            results = []
            for idx in top_indices:
                result = {"content": self.documents[idx], "score": float(similarities[idx])}

                # Add document ID if available
                if idx < len(self.doc_ids):
                    result["id"] = self.doc_ids[idx]

                results.append(result)

            logger.debug(f"Search for '{query}' returned {len(results)} results in {duration:.3f}s")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise QueryError(f"Search failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics about the retriever.

        Returns:
            Dictionary with performance metrics
        """
        stats = self.metrics.get_stats()

        # Add cache statistics
        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / max(1, cache_total)

        stats.update(
            {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "similarity_metric": self.similarity_metric,
            }
        )

        return stats

    def clear(self) -> None:
        """Clear all stored documents and embeddings."""
        self.documents = []
        self.doc_ids = []
        self.embeddings = None
        logger.info("SimpleEmbeddingRetriever cleared all documents and embeddings")


# -----------------------------------------------------------------------------
# Class: ChromaRetriever
# -----------------------------------------------------------------------------
class ChromaRetriever:
    """
    Provides vector database retrieval using ChromaDB for scalable storage and searching.

    Usage:
        retriever = ChromaRetriever(collection_name="my_collection")
        retriever.add_document("Document text", {"category": "demo"}, "doc_1")
        results = retriever.search("demo search query", k=5)

    Enhanced features:
        - Robust error handling
        - Advanced metadata filtering
        - Batch operations
        - Persistence configuration
        - Health check and statistics
    """

    def __init__(
        self,
        collection_name: str = "memories",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ) -> None:
        """
        Initializes a ChromaDB client and obtains (or creates) a collection.

        Args:
            collection_name: Name of the ChromaDB collection (default: "memories").
            persist_directory: Optional directory for persistence (in-memory if None).
            embedding_function: Optional custom embedding function for ChromaDB.
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB not installed. Please install with: pip install chromadb")

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Configure settings based on persistence needs
        try:
            if persist_directory:
                # Create directory if it doesn't exist
                os.makedirs(persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
                logger.info(f"ChromaRetriever using persistent storage at {persist_directory}")
            else:
                # In-memory client
                self.client = chromadb.Client(Settings(allow_reset=True))
                logger.info("ChromaRetriever using in-memory storage")

            # Create or get collection - FIXED: Don't pass embedding_function if None
            # This maintains backward compatibility with the original code
            if embedding_function is not None:
                self.collection = self.client.get_or_create_collection(
                    name=collection_name, embedding_function=embedding_function
                )
            else:
                # Use default ChromaDB behavior (backward compatible)
                self.collection = self.client.get_or_create_collection(name=collection_name)

            # Metrics
            self.metrics = PerformanceMetrics()

        except Exception as e:
            logger.error(f"Failed to initialize ChromaRetriever: {e}")
            raise StorageError(f"ChromaDB initialization error: {e}")

        logger.info(f"ChromaRetriever initialized with collection '{collection_name}'")

    def add_document(self, document: str, metadata: Dict[str, Any], doc_id: str) -> bool:
        """
        Adds a new document to the ChromaDB collection with associated metadata.

        Args:
            document: The text content of the document.
            metadata: Metadata such as categories, tags, etc.
            doc_id: Unique identifier for this document.

        Returns:
            Success status
        """
        if not document or not document.strip():
            logger.warning(f"Empty document provided for doc_id={doc_id}, skipping")
            return False

        try:
            # Convert list values to comma-separated strings for ChromaDB compliance
            processed_metadata: Dict[str, Any] = {}
            for key, val in metadata.items():
                if isinstance(val, list):
                    processed_metadata[key] = ", ".join(map(str, val))
                else:
                    processed_metadata[key] = val

            start_time = time.time()

            # Add to collection
            self.collection.add(documents=[document], metadatas=[processed_metadata], ids=[doc_id])

            # Record metrics
            duration = time.time() - start_time
            self.metrics.log_embedding(duration)

            logger.debug(f"Added doc_id='{doc_id}' to ChromaDB with metadata keys={list(metadata.keys())}")
            return True

        except Exception as e:
            logger.error(f"Error adding document {doc_id} to ChromaDB: {e}")
            raise StorageError(f"Failed to add document to ChromaDB: {e}")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], doc_ids: List[str]) -> bool:
        """
        Add multiple documents to the collection in a single batch operation.

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            doc_ids: List of document IDs

        Returns:
            Success status
        """
        if len(documents) != len(metadatas) or len(documents) != len(doc_ids):
            logger.error("Length mismatch in batch add operation")
            raise ValueError("documents, metadatas, and doc_ids must have the same length")

        if not documents:
            logger.warning("Empty batch provided to add_documents")
            return False

        try:
            # Process metadata for all documents
            processed_metadatas = []
            for metadata in metadatas:
                processed = {}
                for key, val in metadata.items():
                    if isinstance(val, list):
                        processed[key] = ", ".join(map(str, val))
                    else:
                        processed[key] = val
                processed_metadatas.append(processed)

            start_time = time.time()

            # Add batch to collection
            self.collection.add(documents=documents, metadatas=processed_metadatas, ids=doc_ids)

            # Record metrics
            duration = time.time() - start_time
            self.metrics.log_embedding(duration, len(documents))

            logger.debug(f"Added {len(documents)} documents in batch to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error in batch add operation: {e}")
            raise StorageError(f"Failed to add documents in batch: {e}")

    def delete_document(self, doc_id: str) -> bool:
        """
        Removes a document from the collection by ID.

        Args:
            doc_id: ID of the document to delete.

        Returns:
            Success status
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted doc_id='{doc_id}' from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            raise StorageError(f"Failed to delete document from ChromaDB: {e}")

    def update_document(
        self, doc_id: str, document: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Updates an existing document in the collection.

        Args:
            doc_id: ID of the document to update
            document: New document text (or None to keep existing)
            metadata: New metadata (or None to keep existing)

        Returns:
            Success status
        """
        try:
            # First retrieve existing document if needed
            if document is None or metadata is None:
                result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])

                # Check if document exists
                if not result["ids"]:
                    logger.warning(f"Document {doc_id} not found for update")
                    return False

                # Get existing values if needed
                if document is None and result["documents"]:
                    document = result["documents"][0]
                if metadata is None and result["metadatas"]:
                    metadata = result["metadatas"][0]

            # Process metadata if provided
            if metadata:
                processed_metadata: Dict[str, Any] = {}
                for key, val in metadata.items():
                    if isinstance(val, list):
                        processed_metadata[key] = ", ".join(map(str, val))
                    else:
                        processed_metadata[key] = val
            else:
                processed_metadata = {}

            # Update the document
            self.collection.update(
                ids=[doc_id],
                documents=[document] if document else None,
                metadatas=[processed_metadata] if metadata else None,
            )

            logger.debug(f"Updated document {doc_id} in ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise StorageError(f"Failed to update document in ChromaDB: {e}")

    def search(
        self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None, include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Searches for documents similar to the query in the ChromaDB collection.

        Args:
            query: The query text.
            k: Number of top matches to retrieve.
            filter_dict: Optional metadata filters for the query.
            include: Optional list of what to include (["documents", "metadatas", "distances"]).

        Returns:
            Dict[str, Any]: ChromaDB query result structure, typically with:
                - 'documents': List[List[str]]
                - 'metadatas': List[List[Dict[str, Any]]]
                - 'ids': List[List[str]]
                - 'distances': List[List[float]]
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

        start_time = time.time()

        try:
            # Query Chroma
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_dict,
                include=include or ["documents", "metadatas", "distances"],
            )

            # Record metrics
            duration = time.time() - start_time
            self.metrics.log_search(duration)

            # Process 'metadatas' to convert any comma-separated strings back to lists
            if "metadatas" in results and isinstance(results["metadatas"], list):
                for meta_list in results["metadatas"]:
                    if not isinstance(meta_list, list):
                        continue
                    for metadata in meta_list:
                        if not isinstance(metadata, dict):
                            continue
                        for key in ["keywords", "tags"]:
                            val = metadata.get(key)
                            if isinstance(val, str):
                                metadata[key] = [x.strip() for x in val.split(",")]

            logger.debug(f"Search for query='{query}' returned {k} top matches in {duration:.3f}s")
            return results

        except Exception as e:
            logger.error(f"Error during ChromaDB search: {e}")
            raise QueryError(f"ChromaDB search failed: {e}")

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: Document ID to retrieve

        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])

            if not result["ids"]:
                logger.warning(f"Document {doc_id} not found")
                return None

            # Format the result as a single document
            document = {
                "id": result["ids"][0],
                "document": result["documents"][0] if result.get("documents") else None,
                "metadata": result["metadatas"][0] if result.get("metadatas") else {},
            }

            # Process metadata lists
            if document["metadata"] and isinstance(document["metadata"], dict):
                for key in ["keywords", "tags"]:
                    val = document["metadata"].get(key)
                    if isinstance(val, str):
                        document["metadata"][key] = [x.strip() for x in val.split(",")]

            return document

        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            raise StorageError(f"Failed to retrieve document from ChromaDB: {e}")

    def get_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple documents by their IDs.

        Args:
            doc_ids: List of document IDs to retrieve

        Returns:
            List of document data dictionaries
        """
        if not doc_ids:
            return []

        try:
            result = self.collection.get(ids=doc_ids, include=["documents", "metadatas"])

            if not result["ids"]:
                return []

            documents = []
            for i, doc_id in enumerate(result["ids"]):
                doc = {
                    "id": doc_id,
                    "document": result["documents"][i] if result.get("documents") else None,
                    "metadata": result["metadatas"][i] if result.get("metadatas") else {},
                }

                # Process metadata lists
                if doc["metadata"] and isinstance(doc["metadata"], dict):
                    for key in ["keywords", "tags"]:
                        val = doc["metadata"].get(key)
                        if isinstance(val, str):
                            doc["metadata"][key] = [x.strip() for x in val.split(",")]

                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error retrieving multiple documents: {e}")
            raise StorageError(f"Failed to retrieve documents from ChromaDB: {e}")

    def count_documents(self) -> int:
        """
        Count the number of documents in the collection.

        Returns:
            Document count
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {
                "collection_name": self.collection_name,
                "document_count": self.collection.count(),
                "persistent": self.persist_directory is not None,
                "storage_path": self.persist_directory or "in-memory",
            }

            # Add performance metrics
            stats.update(self.metrics.get_stats())

            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"collection_name": self.collection_name, "error": str(e)}

    def check_health(self) -> Dict[str, Any]:
        """
        Perform a health check on the ChromaDB collection.

        Returns:
            Health status information
        """
        health_info = {"status": "unknown", "details": {}}

        try:
            # Check if we can count documents
            count = self.collection.count()
            health_info["document_count"] = count

            # Try a simple query as a health check
            self.collection.query(query_texts=["health check"], n_results=1)

            # Collection is healthy if we get here
            health_info["status"] = "healthy"

            # Check persistence
            if self.persist_directory:
                path = Path(self.persist_directory)
                health_info["details"]["persistent"] = path.exists()
                health_info["details"]["storage_path"] = str(path)
                health_info["details"]["storage_accessible"] = os.access(path, os.W_OK)
            else:
                health_info["details"]["persistent"] = False

            return health_info

        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            health_info["details"]["exception_type"] = type(e).__name__
            logger.error(f"ChromaDB health check failed: {e}")
            return health_info

    def clear(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            Success status
        """
        try:
            # Get all document IDs
            result = self.collection.get(include=[])
            if result["ids"]:
                # Delete all documents
                self.collection.delete(ids=result["ids"])
                logger.info(f"Cleared {len(result['ids'])} documents from ChromaDB collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def query_by_metadata(
        self, metadata_filter: Dict[str, Any], k: int = 100, include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query documents based on metadata filters without needing a text query.

        Args:
            metadata_filter: Filter dictionary for metadata matching
            k: Maximum number of results to return
            include: Optional list of what to include (["documents", "metadatas"])

        Returns:
            Dict containing matching documents
        """
        try:
            results = self.collection.get(where=metadata_filter, limit=k, include=include or ["documents", "metadatas"])

            # Process 'metadatas' to convert any comma-separated strings back to lists
            if "metadatas" in results and isinstance(results["metadatas"], list):
                for i, metadata in enumerate(results["metadatas"]):
                    if not isinstance(metadata, dict):
                        continue
                    for key in ["keywords", "tags"]:
                        val = metadata.get(key)
                        if isinstance(val, str):
                            metadata[key] = [x.strip() for x in val.split(",")]

            logger.debug(f"Metadata query returned {len(results.get('ids', []))} results")
            return results

        except Exception as e:
            logger.error(f"Error in metadata query: {e}")
            raise QueryError(f"Metadata query failed: {e}")

    def advanced_search(
        self,
        query: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
        include: Optional[List[str]] = None,
        reranking_function: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform an advanced search with optional query text, metadata filters, and custom reranking.

        Args:
            query: Optional text query (if None, only metadata filtering is used)
            metadata_filter: Optional metadata filters
            k: Number of top matches to retrieve
            include: Optional list of what to include
            reranking_function: Optional function to rerank results

        Returns:
            List of processed and formatted search results
        """
        try:
            if not query and not metadata_filter:
                logger.warning("No query or metadata filter provided for advanced search")
                return []

            # Initialize empty results
            results = {}

            if query:
                # Text-based search with optional metadata filtering
                results = self.search(
                    query=query,
                    k=k,
                    filter_dict=metadata_filter,
                    include=include or ["documents", "metadatas", "distances"],
                )
            else:
                # Metadata-only filtering
                results = self.query_by_metadata(
                    metadata_filter=metadata_filter, k=k, include=include or ["documents", "metadatas"]
                )

            # Format results into a list of dictionaries
            formatted_results = []

            # Check if we got results
            if not results.get("ids") or not results["ids"]:
                return []

            # Process as a single result pack
            if isinstance(results["ids"], list) and results["ids"]:
                # Handle standard query results where we have inner lists
                if isinstance(results["ids"][0], list):
                    if not results["ids"][0]:
                        return []

                    for i, doc_id in enumerate(results["ids"][0]):
                        result = {
                            "id": doc_id,
                            "content": results["documents"][0][i] if results.get("documents") else None,
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        }

                        # Add distance/score if available
                        if results.get("distances") and results["distances"][0]:
                            result["score"] = float(results["distances"][0][i])

                        formatted_results.append(result)
                # Handle get() results where we have flat lists
                else:
                    for i, doc_id in enumerate(results["ids"]):
                        result = {
                            "id": doc_id,
                            "content": results["documents"][i] if results.get("documents") else None,
                            "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                        }
                        formatted_results.append(result)

            # Apply custom reranking if provided
            if reranking_function and formatted_results:
                formatted_results = reranking_function(formatted_results)

            return formatted_results[:k]  # Ensure we don't exceed the requested number

        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            raise QueryError(f"Advanced search failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics about the retriever.

        Returns:
            Dictionary with performance statistics
        """
        return self.metrics.get_stats()

    def export_collection(self, output_path: str) -> bool:
        """
        Export the entire collection to a JSON file.

        Args:
            output_path: Path to save the JSON export

        Returns:
            Success status
        """
        try:
            # Get all documents with metadata
            result = self.collection.get(include=["documents", "metadatas"])

            if not result["ids"]:
                logger.warning("No documents to export")
                return False

            # Prepare export data
            export_data = {
                "collection_name": self.collection_name,
                "timestamp": datetime.now().isoformat(),
                "document_count": len(result["ids"]),
                "items": [],
            }

            # Process documents
            for i, doc_id in enumerate(result["ids"]):
                item = {
                    "id": doc_id,
                    "document": result["documents"][i] if result.get("documents") else "",
                    "metadata": result["metadatas"][i] if result.get("metadatas") else {},
                }
                export_data["items"].append(item)

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Exported {len(result['ids'])} documents to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting collection: {e}")
            return False

    def import_collection(self, input_path: str, overwrite_existing: bool = False) -> bool:
        """
        Import documents from a JSON file exported with export_collection.

        Args:
            input_path: Path to the JSON file to import
            overwrite_existing: Whether to overwrite documents with existing IDs

        Returns:
            Success status
        """
        try:
            # Load the import file
            with open(input_path, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            if not import_data.get("items"):
                logger.warning("No items found in import file")
                return False

            # Process in batches
            batch_size = 100
            items = import_data["items"]
            total_imported = 0

            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]

                documents = []
                metadatas = []
                ids = []

                for item in batch:
                    doc_id = item["id"]

                    # Check if document already exists
                    if not overwrite_existing:
                        existing = self.get_document(doc_id)
                        if existing:
                            logger.debug(f"Skipping existing document {doc_id}")
                            continue

                    documents.append(item["document"])
                    metadatas.append(item["metadata"])
                    ids.append(doc_id)

                if documents:
                    # Delete existing documents if overwriting
                    if overwrite_existing:
                        for doc_id in ids:
                            try:
                                self.collection.delete(ids=[doc_id])
                            except:
                                pass

                    # Add the batch
                    self.add_documents(documents, metadatas, ids)
                    total_imported += len(documents)

            logger.info(f"Imported {total_imported} documents from {input_path}")
            return True

        except Exception as e:
            logger.error(f"Error importing collection: {e}")
            return False
