from eidosian_core import eidosian

"""
Demonstration of VectorWorker functionality.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from word_forge.database.database_manager import DBManager
from word_forge.vectorizer.vector_store import StorageType, VectorStore
from word_forge.vectorizer.vector_worker import (
    SimpleEmbedder,
    TransformerEmbedder,
    VectorWorker,
)


@eidosian()
@contextmanager
def temporary_database(path: Path) -> Iterator[Path]:
    """
    Create a temporary database for testing.

    Args:
        path: Path where temporary database will be created

    Yields:
        Path to the created database

    Raises:
        sqlite3.Error: If database creation fails
    """
    conn = None
    try:
        # Create the database
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        # Create schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                definition TEXT NOT NULL,
                usage_examples TEXT
            )
            """)

        # Add sample data
        sample_words = [
            (
                1,
                "algorithm",
                "A process or set of rules to be followed for calculations or problem-solving",
                "The sorting algorithm efficiently organized the data; Computer scientists developed a new algorithm for image recognition",
            ),
            (
                2,
                "recursion",
                "The process of defining something in terms of itself",
                "The function uses recursion to calculate factorial; Recursion is a powerful technique in programming",
            ),
        ]

        cursor.executemany("INSERT OR REPLACE INTO words VALUES (?, ?, ?, ?)", sample_words)
        conn.commit()

        # Yield the path to the caller
        yield path

    except sqlite3.Error as e:
        if path.exists():
            path.unlink()
        raise e
    finally:
        if conn:
            conn.close()
        # Clean up the temp file after use
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


@eidosian()
def main() -> None:
    """Demonstrate VectorWorker initialization and operation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("VectorWorkerDemo")

    temp_db_path = Path("./temp_vector_worker_db.sqlite")
    temp_vector_path = Path("./temp_vector_store")

    worker = None
    try:
        with temporary_database(temp_db_path) as db_path:
            logger.info(f"Using temporary database at {db_path}")
            db_manager = DBManager(db_path=str(db_path))

            # Initialize VectorStore (in memory for demo)
            vector_store = VectorStore(
                storage_type=StorageType.MEMORY,
                db_manager=db_manager,
                demo_mode=True,
            )

            # Initialize Embedder (Simple for speed, Transformer if available)
            try:
                embedder = TransformerEmbedder()
                logger.info("Using TransformerEmbedder")
            except Exception:
                logger.warning("TransformerEmbedder failed, using SimpleEmbedder")
                embedder = SimpleEmbedder(dimension=vector_store.dimension)

            # Initialize and start the worker
            worker = VectorWorker(
                db=db_manager,
                vector_store=vector_store,
                embedder=embedder,
                poll_interval=5.0,  # Poll every 5 seconds for demo
                logger=logger,
            )

            logger.info("Starting VectorWorker...")
            worker.start()

            # Let the worker run for a couple of cycles
            time.sleep(12)

            # Check status
            status = worker.get_status()
            logger.info(f"Worker Status: {status}")

            # Verify embeddings were created (check vector store count)
            vector_count = vector_store.collection.count()
            logger.info(f"Vector store contains {vector_count} embeddings.")
            # Expected count depends on sample data (2 words * (term + def + 2 examples)) = 8
            if vector_count > 0:
                logger.info("Embeddings were successfully created and stored.")
            else:
                logger.warning("No embeddings found in the vector store.")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
    finally:
        if worker and worker.is_alive():
            logger.info("Stopping VectorWorker...")
            worker.stop()
            worker.join(timeout=5.0)
            logger.info("VectorWorker stopped.")
        # Clean up vector store directory if it exists
        if temp_vector_path.exists():
            import shutil

            try:
                shutil.rmtree(temp_vector_path)
                logger.info(f"Cleaned up vector store at {temp_vector_path}")
            except OSError as e:
                logger.warning(f"Could not remove vector store directory: {e}")


if __name__ == "__main__":
    main()
