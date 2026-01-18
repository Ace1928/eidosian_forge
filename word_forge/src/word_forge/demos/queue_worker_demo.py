"""
Demonstration of WordProcessor and ParallelWordProcessor functionality.
"""

import logging
import time

from word_forge.database.database_manager import DBManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager
from word_forge.queue.queue_worker import (
    ParallelWordProcessor,
    WordProcessor,
    WorkerPoolConfig,
)


def main() -> None:
    """
    Demonstrate WordProcessor functionality with comprehensive examples.

    This function provides a complete demonstration of:
    - Creating and configuring a word processor
    - Processing individual terms
    - Parallel processing with multiple workers
    - Monitoring performance and relationships
    - Error handling and metrics collection

    The demonstration runs as a complete self-contained example,
    showing the capabilities of the worker processor system.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("WordProcessorDemo")

    logger.info("Starting WordProcessor demonstration")

    # Initialize components
    db_manager = None
    parallel_processor = None

    try:
        # Create database manager with in-memory database for demonstration
        db_manager = DBManager(":memory:")
        db_manager.create_tables()

        # Create queue manager for term management
        queue_manager = QueueManager[str]()

        # Create parser refiner for lexical processing
        parser = ParserRefiner(db_manager, queue_manager)

        # Create word processor
        processor = WordProcessor(db_manager, parser, logger)

        logger.info("Components initialized successfully")

        # ---- Part 1: Process individual terms ----
        logger.info("\n=== Part 1: Individual Term Processing ===")

        # Process a term
        demo_term = "algorithm"
        logger.info(f"Processing term: '{demo_term}'")

        # Insert a sample term directly for demonstration
        db_manager.insert_or_update_word(
            demo_term,
            "A step-by-step procedure for calculations or problem-solving.",
            "noun",
            [
                "The sorting algorithm runs in O(n log n) time.",
                "She developed a new algorithm for image recognition.",
            ],
        )

        # Process the term
        result = processor.process(demo_term)

        if result.is_success:
            logger.info(f"Successfully processed '{demo_term}'")

            # Get statistics
            stats = processor.get_statistics()
            logger.info(f"Processing statistics: {stats}")
        else:
            logger.error(f"Failed to process '{demo_term}': {result.error}")

        # Try processing again (should detect duplicate)
        logger.info(f"Processing '{demo_term}' again (should be duplicate)")
        result = processor.process(demo_term)

        # Get updated statistics
        stats = processor.get_statistics()
        logger.info(f"Updated statistics: {stats}")

        # ---- Part 2: Parallel processing with worker pool ----
        logger.info("\n=== Part 2: Parallel Processing ===")

        # Reset processor statistics
        processor.reset_statistics()

        # Create parallel processor with 3 workers
        pool_config = WorkerPoolConfig(worker_count=3, max_queue_size=100)
        parallel_processor = ParallelWordProcessor(processor, pool_config, logger)

        # Create sample terms
        sample_terms = [
            "recursion",
            "function",
            "compiler",
            "variable",
            "database",
            "interface",
            "programming",
            "software",
            "development",
            "testing",
        ]

        # Add terms to the queue
        for term in sample_terms:
            # Create sample database entries
            db_manager.insert_or_update_word(
                term, f"Definition of {term}", "noun", [f"Example usage of {term}."]
            )

            # Enqueue for processing
            queue_manager.enqueue(term)

        logger.info(f"Enqueued {len(sample_terms)} terms for processing")

        # Start parallel processing
        logger.info("Starting parallel processing")
        parallel_processor.start()

        # Monitor progress
        previous_count = 0
        max_wait = 5  # Max wait in seconds
        wait_start = time.time()

        while True:
            # Get current status
            status = parallel_processor.get_status()
            stats = status["stats"]

            # Check for new processed items
            current_count = stats["processed_count"]
            if current_count > previous_count:
                logger.info(
                    f"Progress: {current_count}/{len(sample_terms)} terms processed "
                    f"({stats['processing_rate_per_minute']:.1f} terms/min)"
                )
                previous_count = current_count
                wait_start = time.time()  # Reset wait timer

            # All terms processed or timeout
            if (
                current_count >= len(sample_terms)
                or time.time() - wait_start > max_wait
            ):
                break

            time.sleep(0.1)

        # Stop parallel processing
        parallel_processor.stop()

        # Show final statistics
        final_status = parallel_processor.get_status()
        final_stats = final_status["stats"]

        logger.info("\n=== Final Processing Statistics ===")
        logger.info(f"Terms processed: {final_stats['processed_count']}")
        logger.info(f"Successful: {final_stats['success_count']}")
        logger.info(f"Duplicates: {final_stats['duplicate_count']}")
        logger.info(f"Errors: {final_stats['error_count']}")
        logger.info(
            f"Average processing time: {final_stats['avg_processing_time_ms']:.2f}ms"
        )
        logger.info(
            f"Processing rate: {final_stats['processing_rate_per_minute']:.2f} terms/min"
        )

        if final_stats["relationship_counts"]:
            logger.info("\nRelationship Types Discovered:")
            for rel_type, count in final_stats["relationship_counts"].items():
                logger.info(f"  - {rel_type}: {count}")

        logger.info("\nWordProcessor demonstration completed successfully!")

    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        try:
            if parallel_processor is not None:
                parallel_processor.stop()
            if db_manager is not None:
                db_manager.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
