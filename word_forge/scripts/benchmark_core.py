#!/usr/bin/env python3
"""
Benchmark Core Components of Word Forge.

This script measures the performance of key components:
1. Database Operations (SQLite)
2. Graph Operations (NetworkX)
3. Vector Store Operations (ChromaDB + Embeddings)

It generates a report with operations per second (OPS) and latency.
"""

import logging
import os
import random
import shutil
import string
import sys
import tempfile
import time
from pathlib import Path

# Ensure we can import word_forge and eidosian_core
sys.path.insert(0, str(Path(__file__).parents[1].resolve() / "src"))
sys.path.insert(0, str(Path(__file__).parents[2].resolve() / "lib"))

# Mock or specific model for benchmarking
from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.vectorizer.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("benchmark")


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def benchmark_database(temp_dir: str, num_ops: int = 1000):
    LOGGER.info(f"--- Benchmarking Database ({num_ops} ops) ---")
    db_path = os.path.join(temp_dir, "bench.db")
    db_manager = DBManager(db_path=db_path)
    db_manager.create_tables()

    # Insert
    start_time = time.time()
    for i in range(num_ops):
        word = f"word_{i}"
        db_manager.execute_query("INSERT INTO words (term, last_refreshed) VALUES (?, ?)", (word, time.time()))
    end_time = time.time()
    duration = end_time - start_time
    ops = num_ops / duration
    LOGGER.info(f"DB Insert: {duration:.4f}s ({ops:.2f} ops/sec)")

    # Read
    start_time = time.time()
    for i in range(num_ops):
        word = f"word_{i}"
        db_manager.get_word_entry(word)
    end_time = time.time()
    duration = end_time - start_time
    ops = num_ops / duration
    LOGGER.info(f"DB Read: {duration:.4f}s ({ops:.2f} ops/sec)")

    db_manager.close()


def benchmark_graph(temp_dir: str, num_nodes: int = 1000, num_edges: int = 2000):
    LOGGER.info(f"--- Benchmarking Graph ({num_nodes} nodes, {num_edges} edges) ---")
    db_path = os.path.join(temp_dir, "graph_bench.db")
    db_manager = DBManager(db_path=db_path)
    db_manager.create_tables()

    graph_manager = GraphManager(db_manager=db_manager)

    # Pre-populate DB so graph builder finds them
    with db_manager.get_connection() as conn:
        for i in range(num_nodes):
            conn.execute("INSERT INTO words (term, last_refreshed) VALUES (?, ?)", (f"node_{i}", time.time()))
        conn.commit()

    # Build Graph (In-memory mostly)
    start_time = time.time()
    # We simulate adding nodes/edges directly to the networkx graph
    # since build_graph reads from DB relationships which are empty.
    # So we'll benchmark the graph structure manipulation itself.

    G = graph_manager.g
    for i in range(num_nodes):
        G.add_node(f"node_{i}", type="word")

    for _ in range(num_edges):
        u = f"node_{random.randint(0, num_nodes-1)}"
        v = f"node_{random.randint(0, num_nodes-1)}"
        if u != v:
            G.add_edge(u, v, weight=random.random())

    end_time = time.time()
    duration = end_time - start_time
    LOGGER.info(f"Graph Build (Direct NX): {duration:.4f}s")

    # Benchmark Analysis (Multidimensional)
    start_time = time.time()
    graph_manager.analysis.analyze_multidimensional_relationships()
    end_time = time.time()
    duration = end_time - start_time
    LOGGER.info(f"Graph Analysis (Multidimensional): {duration:.4f}s")

    db_manager.close()


def benchmark_vector(temp_dir: str, num_docs: int = 50):
    LOGGER.info(f"--- Benchmarking Vector Store ({num_docs} docs) ---")
    # Use a small model for speed, or mock if possible.
    # Real model loading takes time, so we exclude load time from op time if possible,
    # but VectorStore loads in __init__.

    db_path = os.path.join(temp_dir, "vector_bench.db")
    db_manager = DBManager(db_path=db_path)
    db_manager.create_tables()

    # We use a very small model or the default.
    # Warning: This downloads the model if not present.
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    LOGGER.info(f"Loading VectorStore with {model_name}...")
    try:
        start_load = time.time()
        vector_store = VectorStore(db_manager=db_manager, model_name=model_name)
        LOGGER.info(f"Model loaded in {time.time() - start_load:.2f}s")

        docs = [
            f"This is document number {i} with some random content {generate_random_string()}" for i in range(num_docs)
        ]
        metadatas = [{"term": f"doc_{i}", "type": "test"} for i in range(num_docs)]
        ids = [f"doc_{i}" for i in range(num_docs)]

        # Add
        start_time = time.time()
        for i, doc_text in enumerate(docs):
            # Create a mock WordEntryDict
            entry = {
                "id": i,
                "term": f"term_{i}",
                "definition": doc_text,
                "language": "en",
                "usage_examples": [],
                "part_of_speech": "noun",
                "last_refreshed": time.time(),
                "relationships": [],
            }
            vector_store.store_word(entry)

        end_time = time.time()
        duration = end_time - start_time
        ops = num_docs / duration
        LOGGER.info(f"Vector Add: {duration:.4f}s ({ops:.2f} docs/sec)")

        # Search
        start_time = time.time()
        results = vector_store.search(query_text="random content", k=5)
        end_time = time.time()
        duration = end_time - start_time
        LOGGER.info(f"Vector Search: {duration:.4f}s")

    except Exception as e:
        LOGGER.error(f"Vector benchmark failed: {e}")
    finally:
        db_manager.close()


def main():
    temp_dir = tempfile.mkdtemp(prefix="word_forge_bench_")
    LOGGER.info(f"Running benchmarks in {temp_dir}")

    try:
        benchmark_database(temp_dir)
        benchmark_graph(temp_dir)
        benchmark_vector(temp_dir)  # This might be slow due to model load
    finally:
        shutil.rmtree(temp_dir)
        LOGGER.info("Benchmark complete, temp dir cleaned.")


if __name__ == "__main__":
    main()
