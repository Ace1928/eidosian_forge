"""
Demonstration of EmotionWorker functionality.
"""

import logging
import sys
import time

from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.emotion.emotion_processor import RecursiveEmotionProcessor
from word_forge.emotion.emotion_worker import EmotionWorker

# Conditionally import recursive processor
RECURSIVE_PROCESSOR_AVAILABLE = True


def main() -> None:
    """Demonstrate EmotionWorker initialization and operation."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("EmotionWorkerDemo")

    # Initialize dependencies with actual DBManager instead of temporary database
    db_manager = DBManager()

    # Create database tables if they don't exist
    db_manager.create_tables()

    # Initialize emotion manager with VADER and LLM if available
    emotion_mgr = EmotionManager(db_manager)

    # Ensure LLM and VADER are enabled if available
    from word_forge.emotion.emotion_types import LLM_AVAILABLE, VADER_AVAILABLE

    if VADER_AVAILABLE:
        emotion_mgr.config.enable_vader = True
    if LLM_AVAILABLE:
        setattr(emotion_mgr.config, "enable_llm", True)
        setattr(emotion_mgr.config, "llm_weight", 0.6)  # Give significant weight to LLM

    # Re-initialize analysis tools with updated config
    emotion_mgr.init_analysis_tools()

    # Initialize recursive processor if available (for demonstration)
    processor = None
    if RECURSIVE_PROCESSOR_AVAILABLE:
        try:
            processor = RecursiveEmotionProcessor(db_manager, emotion_mgr)
            print("Recursive emotion processor initialized with LLM capabilities")
        except Exception as e:
            print(f"Could not initialize RecursiveEmotionProcessor: {str(e)}")
            print("Using fallback emotion processing methods")

    # Seed the database with some sample words if needed
    sample_words = [
        "happiness",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "trust",
        "anticipation",
        "nostalgia",  # Added for recursive depth
        "melancholy",  # Added for recursive depth
        "contentment",  # Added for recursive depth
    ]
    print("Ensuring sample words exist in database...")
    for word in sample_words:
        try:
            db_manager.insert_or_update_word(word, f"The emotion of {word}", "noun")
            print(f"Added/updated word: {word}")
        except Exception as e:
            print(f"Error adding {word}: {e}")

    # Configure and start the worker
    worker = EmotionWorker(
        db=db_manager,
        emotion_manager=emotion_mgr,
        processor=processor,
        poll_interval=10.0,
        batch_size=20,
        strategy="hybrid" if processor else "random",
        confidence_threshold=0.6,
        enable_logging=True,
    )

    # Function to display worker status
    def display_status():
        status = worker.get_status()
        logger.info("\nEmotion Worker Status:")
        logger.info("-" * 60)
        logger.info(f"Running: {status['running']}")
        logger.info(f"State: {status['state']}")
        logger.info(f"Words processed: {status['processed_count']}")
        logger.info(f"Errors encountered: {status['error_count']}")
        if status["last_update"]:
            last_update = time.strftime(
                "%H:%M:%S", time.localtime(status["last_update"])
            )
            logger.info(f"Last update: {last_update}")
        if status["uptime"]:
            logger.info(f"Uptime: {status['uptime']:.1f} seconds")
        if status["backlog_estimate"] > 0:
            logger.info(f"Estimated backlog: {status['backlog_estimate']} words")
        if status["recent_errors"]:
            logger.info(f"Recent error types: {', '.join(status['recent_errors'])}")
        logger.info(f"Strategy: {status['strategy']}")
        if status["next_poll"]:
            next_poll = time.strftime("%H:%M:%S", time.localtime(status["next_poll"]))
            logger.info(f"Next poll: {next_poll}")
        logger.info("-" * 60)

    try:
        print("Starting emotion worker...")
        worker.start()

        # Determine run duration with default and argument support
        run_seconds = 60  # Default
        if len(sys.argv) > 1:
            try:
                run_seconds = int(sys.argv[1])
                print(
                    f"Will run for {run_seconds} seconds (from command line argument)"
                )
            except ValueError:
                print(f"Invalid duration argument, using default: {run_seconds}s")
        else:
            print(f"Worker will run for {run_seconds} seconds...")

        # Define time thresholds for demonstrations
        pause_time = min(15, run_seconds // 4) if run_seconds > 30 else None
        restart_time = min(30, run_seconds // 2) if run_seconds > 60 else None

        # Main monitoring loop
        start_time = time.time()
        while time.time() - start_time < run_seconds:
            # Show current status every 5 seconds
            display_status()

            # Demonstrate pause functionality if appropriate
            elapsed = time.time() - start_time
            if pause_time and 0.99 * pause_time <= elapsed <= 1.01 * pause_time:
                print("\nDemonstrating pause functionality...")
                worker.pause()
                time.sleep(3)  # Pause for 3 seconds
                print("Resuming worker...")
                worker.resume()

            # Demonstrate restart functionality if appropriate
            if restart_time and 0.99 * restart_time <= elapsed <= 1.01 * restart_time:
                print("\nDemonstrating restart functionality...")
                worker.restart()

            # Wait before next status check
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("Stopping emotion worker...")
        worker.stop()
        worker.join(timeout=5.0)
        print("Worker stopped.")

        # Show final status
        display_status()

        # Display the emotional values assigned to words
        print("\nEmotional values assigned to words:")
        print("-" * 60)
        for word in sample_words:
            try:
                word_id = db_manager.get_word_id(word)
                if word_id:
                    emotion_data = emotion_mgr.get_word_emotion(word_id)
                    if emotion_data:
                        print(
                            f"{word}: valence={emotion_data['valence']:.2f}, "
                            f"arousal={emotion_data['arousal']:.2f}"
                        )
                    else:
                        print(f"{word}: No emotional data assigned")
            except Exception as e:
                print(f"Error retrieving emotion for {word}: {e}")

        # Show performance stats
        if worker.get_status()["processed_count"] > 0:
            print("\nPerformance Statistics:")
            print("-" * 60)
            stats = worker.get_performance_stats()
            if "avg_process_time" in stats:
                print(
                    f"Average batch processing time: {stats['avg_process_time']:.2f}s"
                )
            if "avg_batch_size" in stats:
                print(f"Average batch size: {stats['avg_batch_size']:.1f} words")
            if "words_per_second" in stats:
                print(f"Processing rate: {stats['words_per_second']:.2f} words/second")
            if "estimated_completion" in stats:
                print(
                    f"Estimated time to process backlog: {stats['estimated_completion']:.1f}s"
                )
            if stats["error_summary"]:
                print("Error distribution:")
                for error_type, count in stats["error_summary"].items():
                    print(f"  {error_type}: {count}")


if __name__ == "__main__":
    main()
