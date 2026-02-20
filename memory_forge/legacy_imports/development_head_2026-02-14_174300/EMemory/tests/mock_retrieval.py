"""
🔹 Enhanced Mock Retrieval Components for Testing 🔹

Provides realistic mock implementations of ChromaDB and embedding retriever systems
for use in both unit and integration tests.

Features:
- Full ChromaDB API mock implementation
- Performance simulation with configurable delays
- Error injection capabilities
- Call history tracking for test assertions
- Realistic document scoring and ranking
"""

import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MockRetrievalError(Exception):
    """Base exception for mock retrieval errors."""

    pass


class MockChromaCollection:
    """Mock implementation of ChromaDB collection for testing."""

    def __init__(self, name: str = "test_collection"):
        self.name = name
        self.data: Dict[str, Dict[str, Any]] = {}
        self.call_history: List[Dict[str, Any]] = []
        self._should_fail: Dict[str, bool] = defaultdict(bool)
        self._delays: Dict[str, float] = defaultdict(float)

    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> None:
        """Mock adding documents to collection."""
        self._simulate_operation("add")
        self._record_call("add", {"documents": documents, "metadatas": metadatas, "ids": ids})

        for i, doc_id in enumerate(ids):
            self.data[doc_id] = {
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Mock retrieving documents by ID or filter."""
        self._simulate_operation("get")
        self._record_call("get", {"ids": ids, "where": where, "limit": limit, "include": include})

        # Default includes
        if include is None:
            include = ["documents", "metadatas", "embeddings"]

        # Filter by IDs if provided
        filtered_ids = ids if ids else list(self.data.keys())

        # Apply metadata filter if specified
        if where:
            filtered_ids = [
                doc_id for doc_id in filtered_ids if self._matches_filter(self.data[doc_id]["metadata"], where)
            ]

        # Apply limit if specified
        if limit is not None and limit < len(filtered_ids):
            filtered_ids = filtered_ids[:limit]

        # Prepare result
        result = {"ids": filtered_ids}

        if "documents" in include:
            result["documents"] = [self.data[doc_id]["document"] for doc_id in filtered_ids]

        if "metadatas" in include:
            result["metadatas"] = [self.data[doc_id]["metadata"] for doc_id in filtered_ids]

        if "embeddings" in include:
            # Generate mock embeddings
            result["embeddings"] = [[0.1, 0.2, 0.3] for _ in filtered_ids]

        return result

    def query(
        self,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Mock querying for similar documents."""
        self._simulate_operation("query")
        self._record_call(
            "query", {"query_texts": query_texts, "n_results": n_results, "where": where, "include": include}
        )

        # Default includes
        if include is None:
            include = ["documents", "metadatas", "distances"]

        # Get all IDs (applying filter if specified)
        all_ids = list(self.data.keys())
        if where:
            all_ids = [doc_id for doc_id in all_ids if self._matches_filter(self.data[doc_id]["metadata"], where)]

        # Add some randomization but keep order mostly consistent
        if query_texts and query_texts[0]:
            # Use query as random seed for reproducible "relevance" ordering
            seed = hash(query_texts[0]) % 10000
            random.seed(seed)
            # Shuffle slightly to simulate semantic search
            random.shuffle(all_ids)

        # Limit results
        result_ids = all_ids[:n_results] if n_results < len(all_ids) else all_ids

        # Prepare result (ChromaDB returns nested lists for query results)
        result = {"ids": [result_ids]}

        if "documents" in include:
            result["documents"] = [[self.data[doc_id]["document"] for doc_id in result_ids]]

        if "metadatas" in include:
            result["metadatas"] = [[self.data[doc_id]["metadata"] for doc_id in result_ids]]

        if "distances" in include:
            # Generate mock distances (higher index = lower relevance)
            # Add a small random factor to seem more realistic
            distances = []
            for i in range(len(result_ids)):
                base_score = 0.99 - (0.1 * i)
                jitter = random.uniform(-0.05, 0.05)
                distances.append(max(0.01, min(0.99, base_score + jitter)))
            result["distances"] = [distances]

        return result

    def delete(self, ids: List[str]) -> None:
        """Mock deleting documents by ID."""
        self._simulate_operation("delete")
        self._record_call("delete", {"ids": ids})

        for doc_id in ids:
            if doc_id in self.data:
                del self.data[doc_id]

    def update(
        self, ids: List[str], documents: Optional[List[str]] = None, metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Mock updating documents."""
        self._simulate_operation("update")
        self._record_call("update", {"ids": ids, "documents": documents, "metadatas": metadatas})

        for i, doc_id in enumerate(ids):
            if doc_id in self.data:
                if documents and i < len(documents):
                    self.data[doc_id]["document"] = documents[i]
                if metadatas and i < len(metadatas):
                    self.data[doc_id]["metadata"] = metadatas[i]

    def upsert(self, ids: List[str], documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Mock upserting documents (add if not exists, update if exists)."""
        self._simulate_operation("upsert")
        self._record_call("upsert", {"ids": ids, "documents": documents, "metadatas": metadatas})

        for i, doc_id in enumerate(ids):
            if i < len(documents):
                doc = documents[i]
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}

                if doc_id in self.data:
                    # Update
                    self.data[doc_id]["document"] = doc
                    self.data[doc_id]["metadata"] = meta
                else:
                    # Insert
                    self.data[doc_id] = {"document": doc, "metadata": meta}

    def count(self) -> int:
        """Return the number of documents in the collection."""
        self._simulate_operation("count")
        self._record_call("count", {})
        return len(self.data)

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            # Handle different types of comparisons
            if isinstance(value, dict):
                # Handle operators like $gt, $lt, etc.
                for op, op_val in value.items():
                    if op == "$eq" and metadata[key] != op_val:
                        return False
                    elif op == "$ne" and metadata[key] == op_val:
                        return False
                    elif op == "$gt" and not (metadata[key] > op_val):
                        return False
                    elif op == "$lt" and not (metadata[key] < op_val):
                        return False
                    elif op == "$gte" and not (metadata[key] >= op_val):
                        return False
                    elif op == "$lte" and not (metadata[key] <= op_val):
                        return False
                    elif op == "$in" and metadata[key] not in op_val:
                        return False
                    elif op == "$nin" and metadata[key] in op_val:
                        return False
            elif metadata[key] != value:
                return False

        return True

    def _record_call(self, method: str, params: Dict[str, Any]) -> None:
        """Record method calls for inspection in tests."""
        self.call_history.append({"method": method, "params": params, "timestamp": time.time()})

    def _simulate_operation(self, operation: str) -> None:
        """Simulate delays and errors for the specified operation."""
        # Apply delay if configured
        if operation in self._delays and self._delays[operation] > 0:
            time.sleep(self._delays[operation])

        # Raise error if configured
        if operation in self._should_fail and self._should_fail[operation]:
            raise MockRetrievalError(f"Simulated failure in {operation}")

    def set_delay(self, operation: str, seconds: float) -> None:
        """Configure a delay for the specified operation."""
        self._delays[operation] = seconds

    def set_error(self, operation: str, should_fail: bool = True) -> None:
        """Configure whether the specified operation should fail."""
        self._should_fail[operation] = should_fail

    def reset(self) -> None:
        """Clear all data and call history."""
        self.data = {}
        self.call_history = []
        self._should_fail = defaultdict(bool)
        self._delays = defaultdict(float)


class MockChromaClient:
    """Mock implementation of ChromaDB client."""

    def __init__(self):
        self.collections: Dict[str, MockChromaCollection] = {}
        self.call_history: List[Dict[str, Any]] = []

    def get_or_create_collection(self, name: str, **kwargs) -> MockChromaCollection:
        """Get or create a mock collection."""
        self._record_call("get_or_create_collection", {"name": name, **kwargs})
        if name not in self.collections:
            self.collections[name] = MockChromaCollection(name)
        return self.collections[name]

    def get_collection(self, name: str) -> Optional[MockChromaCollection]:
        """Get an existing collection."""
        self._record_call("get_collection", {"name": name})
        return self.collections.get(name)

    def list_collections(self) -> List[Dict[str, str]]:
        """List all collections."""
        self._record_call("list_collections", {})
        return [{"name": name} for name in self.collections.keys()]

    def create_collection(self, name: str, **kwargs) -> MockChromaCollection:
        """Create a new collection."""
        self._record_call("create_collection", {"name": name, **kwargs})
        if name in self.collections:
            raise ValueError(f"Collection {name} already exists")
        self.collections[name] = MockChromaCollection(name)
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        self._record_call("delete_collection", {"name": name})
        if name in self.collections:
            del self.collections[name]

    def heartbeat(self) -> int:
        """Return a mock heartbeat for the client."""
        self._record_call("heartbeat", {})
        return int(time.time())

    def _record_call(self, method: str, params: Dict[str, Any]) -> None:
        """Record method calls for inspection in tests."""
        self.call_history.append({"method": method, "params": params, "timestamp": time.time()})

    def reset(self) -> None:
        """Reset all collections and call history."""
        self.collections = {}
        self.call_history = []


class MockSimpleEmbeddingRetriever:
    """Mock implementation of SimpleEmbeddingRetriever for testing."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.documents: List[str] = []
        self.call_history: List[Dict[str, Any]] = []
        self._should_fail = False
        self._delay = 0.0

    def add_document(self, document: str) -> None:
        """Mock adding a document."""
        self._simulate_operation()
        self._record_call("add_document", {"document": document})
        self.documents.append(document)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock searching for documents."""
        self._simulate_operation()
        self._record_call("search", {"query": query, "top_k": top_k})

        # Return up to top_k documents with mock scores
        result_count = min(top_k, len(self.documents))
        results = []

        # Score documents based on simple text matching
        if query and self.documents:
            # Generate scores with a bit of randomness but keep consistent for same query
            random.seed(hash(query) % 10000)

            # Basic relevance scoring with simple text matching and randomness
            scored_docs = []
            for i, doc in enumerate(self.documents):
                # Base score on position (older docs get lower base score)
                base_score = 0.9 - (0.01 * i)

                # Boost score if query terms appear in document
                query_terms = query.lower().split()
                doc_text = doc.lower()

                term_matches = sum(1 for term in query_terms if term in doc_text)
                boost = min(0.5, term_matches * 0.1)

                # Add small random jitter
                jitter = random.uniform(-0.05, 0.05)

                final_score = min(0.99, base_score + boost + jitter)
                scored_docs.append((doc, final_score))

            # Sort by score descending
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Take top k
            for i in range(min(top_k, len(scored_docs))):
                doc, score = scored_docs[i]
                results.append({"content": doc, "score": score})
        else:
            # Default behavior when query is empty or no documents
            for i in range(result_count):
                results.append({"content": self.documents[i], "score": 0.95 - (i * 0.05)})  # Decreasing scores

        return results

    def set_delay(self, seconds: float) -> None:
        """Set delay for operations."""
        self._delay = seconds

    def set_error(self, should_fail: bool = True) -> None:
        """Set whether operations should fail."""
        self._should_fail = should_fail

    def _simulate_operation(self) -> None:
        """Simulate delay and error if configured."""
        if self._delay > 0:
            time.sleep(self._delay)

        if self._should_fail:
            raise MockRetrievalError("Simulated retriever failure")

    def _record_call(self, method: str, params: Dict[str, Any]) -> None:
        """Record method calls for inspection in tests."""
        self.call_history.append({"method": method, "params": params, "timestamp": time.time()})

    def reset(self) -> None:
        """Clear all documents and call history."""
        self.documents = []
        self.call_history = []
        self._should_fail = False
        self._delay = 0.0


class MockMemoryUnit:
    """Factory for creating mock AgenticMemoryUnit objects for testing."""

    @staticmethod
    def create(
        content: str = "Test content",
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        context: str = "Test",
        category: str = "Test",
        tags: Optional[List[str]] = None,
        relationships: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        evolution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Create a dictionary representing a memory unit for testing.

        Args:
            content: Content text
            id: Optional ID (generated if None)
            keywords: Optional list of keywords
            context: Context string
            category: Category string
            tags: Optional list of tags
            relationships: Optional list of related unit IDs
            timestamp: Optional timestamp string (generated if None)
            evolution_history: Optional evolution history records

        Returns:
            Dictionary representing a memory unit
        """
        from datetime import datetime
        from uuid import uuid4

        current_timestamp = datetime.now().strftime("%Y%m%d%H%M")

        return {
            "id": id or str(uuid4()),
            "content": content,
            "keywords": keywords or ["test", "mock"],
            "links": {},
            "retrieval_count": 0,
            "timestamp": timestamp or current_timestamp,
            "last_accessed": current_timestamp,
            "context": context,
            "evolution_history": evolution_history or [],
            "category": category,
            "tags": tags or ["test"],
            "relationships": relationships or [],
        }

    @staticmethod
    def batch_create(count: int, content_prefix: str = "Test content") -> List[Dict[str, Any]]:
        """
        Create multiple mock memory units at once.

        Args:
            count: Number of units to create
            content_prefix: Prefix for content text

        Returns:
            List of memory unit dictionaries
        """
        return [MockMemoryUnit.create(content=f"{content_prefix} {i+1}") for i in range(count)]
