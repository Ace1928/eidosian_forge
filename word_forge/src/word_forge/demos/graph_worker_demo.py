"""
Demonstration of GraphWorker functionality.
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

from word_forge.config import config
from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.graph.graph_worker import GraphWorker


def main() -> None:
    """
    Demonstrate usage of the GraphWorker to maintain a lexical graph.

    Creates a worker thread that periodically updates the graph and saves
    both a GEXF file for storage and an HTML visualization for viewing.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    logger = logging.getLogger("graph_worker_demo")
    logger.info("Starting graph worker demonstration")

    # Initialize database and graph managers
    db_path = config.database.db_path
    logger.info(f"Using database: {db_path}")

    worker = None
    try:
        db_manager = DBManager(db_path=db_path)
        graph_manager = GraphManager(db_manager)

        # Ensure DB has tables and sample data
        db_manager.create_tables()  # Note: using protected method for compatibility
        if graph_manager.ensure_sample_data():
            logger.info("Added sample data to database")

        # Set up paths for the worker
        export_path = Path(config.graph.default_export_path) / "lexical_graph.gexf"
        vis_path = Path(config.graph.visualization_path) / "lexical_graph.html"

        # Pre-create directories to avoid race conditions
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)

        # Configure and start the worker
        worker = GraphWorker(
            graph_manager=graph_manager,
            poll_interval=10.0,  # More frequent updates for demo
            output_path=str(export_path),
            visualization_path=str(vis_path),
        )

        logger.info("Starting graph worker")
        worker.start()

        # Check status periodically
        for _ in range(3):
            time.sleep(5)
            status = worker.get_status()
            logger.info(f"Worker status: {status}")

            # If in error state, show detailed error
            if status["state"] == "error" and status["last_error"]:
                logger.info(f"Error details: {status['last_error']}")

        # Stop the worker
        logger.info("Stopping worker")
        worker.stop()
        worker.join(timeout=15.0)

        final_status = worker.get_status()
        logger.info(f"Final worker status: {final_status}")
        logger.info(
            f"Graph saved to {worker.output_path}, "
            f"visualization available at {worker.visualization_path}"
        )

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        logger.debug(traceback.format_exc())
    finally:
        # Ensure worker is properly stopped if an exception occurs
        if worker and worker.is_alive():
            logger.info("Stopping worker due to exception")
            worker.stop()
            worker.join(timeout=5.0)


if __name__ == "__main__":
    main()
