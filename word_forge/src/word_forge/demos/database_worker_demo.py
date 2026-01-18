"""
Demonstration of DatabaseWorker functionality.
"""

import logging
import sys
import time
from pathlib import Path

from word_forge.database.database_manager import DBManager
from word_forge.database.database_worker import DatabaseWorker


def main() -> None:
    """
    Demonstrate DatabaseWorker initialization and operation.

    This function provides a comprehensive demonstration of:
    - Creating and configuring a database worker
    - Running manual and scheduled operations
    - Monitoring worker status and metrics
    - Error handling and recovery
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    logger = logging.getLogger("DatabaseWorkerDemo")
    logger.info("Starting DatabaseWorker demonstration")

    # Create temp directory for demo
    temp_dir = Path("./db_worker_demo")
    temp_dir.mkdir(exist_ok=True)
    db_path = temp_dir / "test_database.sqlite"
    backup_path = temp_dir / "backups"

    logger.info(f"Using database at {db_path}")

    # Initialize database manager
    db_manager = DBManager(db_path=str(db_path))

    # Create database schema
    db_manager.create_tables()

    # Add some sample data
    sample_words = [
        ("algorithm", "A step-by-step procedure for calculations", "noun"),
        ("data", "Facts and statistics collected for reference", "noun"),
        ("iteration", "The act of repeating a process", "noun"),
        ("variable", "A quantity that can change its value", "noun"),
        ("function", "A relationship or expression involving variables", "noun"),
    ]

    logger.info("Adding sample data to database")
    for term, definition, pos in sample_words:
        db_manager.insert_or_update_word(term, definition, pos)

    # Initialize database worker with short intervals for demonstration
    worker = DatabaseWorker(
        db_manager=db_manager,
        poll_interval=15.0,  # Every 15 seconds for demo
        backup_interval=60.0,  # Every 60 seconds for demo
        optimization_interval=30.0,  # Every 30 seconds for demo
        backup_path=backup_path,
    )

    # Define function to display worker status
    def display_status():
        status = worker.get_status()

        logger.info("\nDatabase Worker Status:")
        logger.info("-" * 60)
        logger.info(f"Running: {status['running']}")
        logger.info(f"State: {status['state']}")
        logger.info(f"Operations completed: {status['operation_count']}")
        logger.info(f"Errors encountered: {status['error_count']}")

        if status["last_operation"]:
            logger.info(f"Last operation: {status['last_operation']}")

        if status["last_update"]:
            logger.info(f"Last update: {time.ctime(status['last_update'])}")

        if status["next_maintenance"]:
            logger.info(f"Next maintenance: {time.ctime(status['next_maintenance'])}")

        if status["integrity_status"]:
            logger.info(f"Integrity status: {status['integrity_status']}")

        if status["pending_operations"] > 0:
            logger.info(f"Pending operations: {status['pending_operations']}")

        if status["recent_errors"]:
            logger.info(f"Recent errors: {', '.join(status['recent_errors'])}")

        logger.info("-" * 60)

    try:
        # Start the worker thread
        logger.info("Starting database worker")
        worker.start()

        # Display initial status
        time.sleep(2)
        display_status()

        # Run a manual maintenance operation
        logger.info("\nRunning manual maintenance operation...")
        worker.run_maintenance(wait=True)

        # Display status after maintenance
        display_status()

        # Demonstrate integrity check
        logger.info("\nRunning database integrity check...")
        is_ok, integrity_status = worker.run_integrity_check(wait=True)
        logger.info(f"Integrity check result: {integrity_status} (passed: {is_ok})")

        # Demonstrate backup
        logger.info("\nRunning manual backup operation...")
        worker.run_backup(wait=True)

        # Display status after operations
        display_status()

        # Pause the worker
        logger.info("\nPausing worker...")
        worker.pause()
        time.sleep(2)

        # Display status while paused
        display_status()

        # Resume the worker
        logger.info("\nResuming worker...")
        worker.resume()

        # Run optimization with higher level
        logger.info("\nRunning optimization level 2...")
        worker.run_optimization(level=2, wait=True)

        # Get metrics
        metrics = worker.get_metrics()
        logger.info("\nWorker Metrics Summary:")
        logger.info("-" * 60)
        logger.info(f"Total operations: {metrics['operation_count']}")
        logger.info(f"Average duration: {metrics['avg_duration_ms']:.2f}ms")

        if metrics["maintenance"]["last_run"]:
            logger.info(
                f"Last maintenance: {time.ctime(metrics['maintenance']['last_run'])}"
            )

        if metrics["backup"]["last_run"]:
            logger.info(f"Last backup: {time.ctime(metrics['backup']['last_run'])}")
            logger.info(f"Backup location: {metrics['backup']['backup_path']}")

        if metrics["optimization"]["last_run"]:
            logger.info(
                f"Last optimization: {time.ctime(metrics['optimization']['last_run'])}"
            )
            logger.info(
                f"Optimization level: {metrics['optimization']['current_level']}"
            )

        logger.info("-" * 60)

        # Wait for a scheduled operation to occur
        logger.info("\nWaiting for scheduled operations...")
        time.sleep(20)

        # Display final status
        display_status()

    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received")
    finally:
        # Clean shutdown
        logger.info("Stopping worker...")
        worker.stop()

        # Wait for worker to terminate
        worker.join(timeout=5.0)
        logger.info("Worker stopped.")

        # Final status
        final_status = worker.get_status()
        logger.info(f"Final operation count: {final_status['operation_count']}")
        logger.info(f"Final error count: {final_status['error_count']}")

        # Show backup files if any were created
        if backup_path.exists():
            backups = list(backup_path.glob("*.backup"))
            if backups:
                logger.info(f"Created {len(backups)} backup(s):")
                for backup in backups:
                    logger.info(f"  - {backup.name} ({backup.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
