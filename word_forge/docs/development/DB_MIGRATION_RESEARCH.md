# Word Forge: Database Migration Research

## 1. Objective
Evolve the Word Forge persistence layer beyond SQLite to support:
- Complex multi-relational graph queries.
- Scalable vector similarity search.
- High concurrency for background ingestion workers.
- ACID properties for multi-turn lexical updates.

## 2. Candidates

### 2.1 PostgreSQL + pgvector (Multi-model)
- **Pros**:
    - Industry standard, extremely reliable.
    - `pgvector` provides integrated, high-performance vector search.
    - Excellent support for full-text search (GIN/GiST indexes).
    - Can handle both relational lexical data and vector embeddings in one system.
    - CTEs (Common Table Expressions) allow for recursive graph traversal (though not as native as Cypher).
- **Cons**:
    - Heavier resource footprint than SQLite.
    - Graph queries can become verbose for deep traversals.
- **Termux Support**: Available via `pkg install postgresql`.

### 2.2 Neo4j (Graph Native)
- **Pros**:
    - Native graph storage and processing.
    - Cypher query language is optimized for relationship-heavy lexical networks.
    - Powerful graph algorithms (PageRank, Leiden, Community Detection) available out-of-the-box.
- **Cons**:
    - Java-based (JVM dependency), which can be extremely heavy on Termux/Android.
    - Integration with vector search often requires external plugins or another DB.
- **Termux Support**: Difficult (requires JVM and significant configuration).

### 2.3 ArangoDB (Multi-model Graph)
- **Pros**:
    - Native support for Key/Value, Document, and Graph models.
    - AQL (ArangoDB Query Language) is expressive and flexible.
    - Integrated search engine (ArangoSearch).
- **Cons**:
    - Less community support than Postgres/Neo4j.
    - Setup on Termux is not standard.

## 3. Recommendation: PostgreSQL + pgvector

**Rational**:
1. **Convergence**: Postgres allows us to unify the `words`, `phrases`, `phonetics`, and `vector` storage into a single, highly-available engine.
2. **Resource Efficiency**: Postgres has a significantly lower overhead than JVM-based Neo4j on Android hardware.
3. **Graph Capability**: While not a native graph DB, Postgres recursive queries and `ltree` or JSONB support are sufficient for lexical hierarchies.
4. **Tooling**: SQLAlchemy/psycopg2 integration is mature and supports the `Result` pattern easily.

## 4. Next Steps
1. Create a proof-of-concept migration script from SQLite to Postgres.
2. Implement a `PostgresDBManager` following the Repository Pattern.
3. Benchmark traversal performance vs current NetworkX/SQLite implementation.
