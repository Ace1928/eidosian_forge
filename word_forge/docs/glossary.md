# Word Forge Glossary

This living glossary defines key terms used throughout the project. New entries should remain in **alphabetical order** to make lookup simple.

> **Maintenance**: Add new terms as they arise. Each term should include a brief definition and, where helpful, references to relevant files or modules.

---

## A

- **Arousal** — Emotional intensity dimension ranging from calm (0.0) to excited (1.0). Used alongside valence for dimensional emotion analysis. See `emotion/emotion_config.py`.

- **Adapter (LoRA/PEFT)** — Parameter-efficient fine-tuning technique that adds small trainable matrices to frozen base models, enabling incremental learning without overwriting core knowledge.

## B

- **Backpressure** — Flow control mechanism that slows producers when consumers can't keep up. Prevents queue overflow and resource exhaustion.

- **Big-O Complexity** — Notation describing algorithm efficiency. Key operations: O(1) average for hash lookups, O(log N) for tree operations, O(N) for full scans.

## C

- **ChromaDB** — Vector database used for embedding storage and similarity search. Provides persistent storage with efficient nearest-neighbor queries. See `vectorizer/vector_store.py`.

- **Circuit Breaker** — Pattern for preventing cascading failures by temporarily stopping requests to failing services. Trips after threshold failures, resets after timeout.

- **Configuration Component** — Dataclass implementing the `ConfigComponent` protocol, providing type-safe settings for a specific subsystem. See `configs/config_essentials.py`.

- **Conversation Manager** — Orchestrates multi-step conversation sessions using several language models and persists messages to the database. See `conversation/conversation_manager.py`.

- **Conversation Worker** — Background thread that processes conversation tasks from a queue, handling message generation and context tracking. See `conversation/conversation_worker.py`.

- **Continual Learning** — Machine learning approach where models update incrementally from new data without forgetting previous knowledge.

## D

- **DBManager** — Core class handling all database operations including connection pooling, transactions, and schema management. See `database/database_manager.py`.

- **Dead Letter Queue (DLQ)** — Queue for messages that fail processing repeatedly. Enables debugging and prevents data loss.

- **Dependency Injection** — Design pattern where dependencies are passed to components rather than created internally. Improves testability and flexibility.

- **Dimensional Analysis** — Emotion analysis using continuous dimensions (valence/arousal) rather than discrete categories.

## E

- **Eidosian** — Design philosophy emphasizing type safety, clear layering, self-documenting code, and continuous self‑improvement. Core principle of Word Forge.

- **Emotion Manager** — Unified interface for emotional processing, integrating VADER, TextBlob, and optional LLM analysis. See `emotion/emotion_manager.py`.

- **Emotion Vector** — Mathematical representation of emotional state in multidimensional space (valence, arousal, dominance).

- **Experience Replay** — Technique storing past interactions in a buffer, sampling from it during training to prevent catastrophic forgetting.

## F

- **FAISS** — Facebook AI Similarity Search library used as an alternative vector index backend. Efficient for large-scale nearest neighbor search.

- **Force-Directed Layout** — Graph visualization algorithm using physics simulation (spring forces, repulsion) to arrange nodes naturally.

## G

- **Graph Builder** — Component that constructs the NetworkX graph from database relationships. See `graph/graph_builder.py`.

- **Graph Manager** — Central orchestrator that builds the semantic network from lexical and emotional data using NetworkX. See `graph/graph_manager.py`.

- **Graph Worker** — Background thread that keeps the lexical graph up to date and saves periodic snapshots. See `graph/graph_worker.py`.

## H

- **Hybrid Search** — Combining vector similarity search with traditional keyword (BM25) search for improved retrieval accuracy.

- **Hypernym** — Word with broader meaning (e.g., "animal" is hypernym of "dog"). Opposite of hyponym.

- **Hyponym** — Word with narrower meaning (e.g., "dog" is hyponym of "animal"). Opposite of hypernym.

## I

- **Incremental Update** — Updating only changed elements rather than rebuilding entirely. Critical for performance with large datasets.

## L

- **Lemmatization** — Reducing words to their base form (lemma), considering part of speech. More accurate than stemming.

- **Lexical Data** — Structured information about words including definitions, usage examples, and relationships.

- **Living Lexicon** — Self-referential, continuously evolving knowledge base that discovers and refines terms recursively.

- **LoRA (Low-Rank Adaptation)** — Parameter-efficient fine-tuning method adding low-rank matrices to transformer layers.

- **LRU Cache** — Least Recently Used cache pattern for memoizing expensive function calls. Evicts oldest entries when full.

## M

- **Meta-Emotion** — Emotional response to an emotion (e.g., feeling guilty about feeling angry). Supported by recursive emotion processing.

- **Model State** — Singleton managing language model lifecycle. Target for refactoring to dependency injection.

## N

- **NetworkX** — Python library used for graph operations and network analysis. Core of Word Forge's semantic graph.

- **NumPy** — Array library providing efficient numeric computation used by vector features.

## O

- **Optimistic Locking** — Concurrency control checking version numbers before updates, failing if data changed since read.

## P

- **Parser Config** — Configuration dataclass for lexical data parser settings including resource paths, model settings, and data source management. See `parser/parser_config.py`.

- **Parser Refiner** — Main parsing pipeline that extracts lexical entries from text and enriches them with relationships. See `parser/parser_refiner.py`.

- **PEFT (Parameter-Efficient Fine-Tuning)** — Family of techniques for adapting large models with minimal parameter updates.

- **Priority Queue** — Queue data structure where items are processed based on priority level rather than arrival order.

- **Protocol** — Python typing construct defining structural interface without inheritance. Preferred over ABC for flexibility.

## Q

- **Queue Manager** — Thread-safe task queue coordinating asynchronous worker threads with priority support. See `queue/queue_manager.py`.

- **Queue Processor** — Protocol defining the interface for components that process items from the queue.

## R

- **Relationship** — Connection between words (synonyms, antonyms, hypernyms, etc.) stored in the database and graph.

- **Repository Pattern** — Abstraction layer between domain logic and data access, enabling database independence and easier testing.

- **Result Pattern** — Monadic error handling pattern that avoids exceptions for cross-component error propagation. Returns `Result[T]` with success value or error.

## S

- **Sentence Transformer** — Neural network model for generating dense vector embeddings from text. Used for semantic search.

- **SQLite** — Embedded relational database used for persistent storage. Single-file, serverless, zero-configuration.

- **Stemming** — Reducing words to root form by removing suffixes. Faster but less accurate than lemmatization.

- **Synset** — WordNet concept grouping related word senses. Each synset has unique ID and definition.

## T

- **Term Extractor** — NLP component that discovers and extracts significant terms from text content. See `parser/parser_refiner.py`.

- **TextBlob** — Python library for processing textual data, providing sentiment analysis and NLP utilities.

- **Transaction** — Database operation unit that ensures atomicity and consistency. All changes commit together or roll back.

- **TypedDict** — Python typing construct for dictionaries with specific string keys and typed values.

## V

- **VADER** — Valence Aware Dictionary and sEntiment Reasoner, rule-based sentiment analysis tool optimized for social media text.

- **Valence** — Emotional pleasantness dimension ranging from negative (-1.0) to positive (1.0).

- **VectorError** — Base exception for vector operations, including storage, search, and embedding generation failures. See `exceptions.py`.

- **Vector Store** — Component providing persistent storage and similarity search for vector embeddings. See `vectorizer/vector_store.py`.

- **Vector Worker** — Background thread that generates embeddings for new or updated content and inserts them into the vector store. See `vectorizer/vector_worker.py`.

## W

- **WAL Mode** — Write-Ahead Logging mode for SQLite enabling concurrent reads during writes. Improves concurrency.

- **Word Forge** — The toolkit for building and exploring a semantic network of terms with vector search and emotion analysis.

- **WordNet** — Lexical database from Princeton that supplies synonyms, definitions, and sense relationships.

- **Worker Manager** — Utility orchestrating multiple background workers such as vector and graph processors. See `queue/worker_manager.py`.

- **Worker Pool** — Collection of worker threads processing items in parallel with shared queue.

---

## Adding New Terms

When adding new terms:

1. Place in alphabetical order within appropriate letter section
2. Start with term in **bold**
3. Use em-dash (—) after term
4. Keep definition concise (1-2 sentences)
5. Add file reference with `See` if applicable
6. Consider whether related terms should be cross-referenced
