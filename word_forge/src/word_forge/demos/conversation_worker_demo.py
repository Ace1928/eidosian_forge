from eidosian_core import eidosian

"""
Demonstration of ConversationWorker functionality.

This script showcases the setup, execution, monitoring, and shutdown
of a ConversationWorker instance, simulating a basic conversation
processing workflow. It integrates various Word Forge components
like database, emotion, queue, and conversation managers.
"""

import logging
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Word Forge Components
from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.conversation.conversation_models import (
    AffectiveLexicalLanguageModel,
    EidosianIdentityModel,
    LightweightLanguageModel,
    ReflexiveLanguageModel,
)
from word_forge.conversation.conversation_worker import ConversationWorker
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.parser.language_model import ModelState
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager
from word_forge.vectorizer.vector_store import VectorStore

# Eidosian Type Alias for clarity and precision
QueueItem = Union["ConversationTask", str, Dict[str, Any]]
ConversationTask = Dict[str, Any]

# Constants for configuration
DB_PATH = "word_forge_demo.sqlite"
WORKER_POLL_INTERVAL = 1.0  # seconds
WORKER_PROCESSING_TIMEOUT = 10.0  # seconds
WORKER_BATCH_SIZE = 5
MONITORING_INTERVAL = 2.0  # seconds
MONITORING_ITERATIONS = 5
SHUTDOWN_TIMEOUT = 5.0  # seconds

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("conversation_worker_demo")


def _create_conversation_task(conversation_id: int, speaker: str, message: str, priority: int = 1) -> ConversationTask:
    """
    Creates a standardized ConversationTask dictionary.

    Args:
        conversation_id: The ID of the conversation this task belongs to.
        speaker: The identifier of the speaker (e.g., 'user', 'agent').
        message: The content of the message.
        priority: The priority level of the task (default: 1).

    Returns:
        A dictionary representing the ConversationTask, adhering to the
        expected structure, including a unique task ID and timestamp.
        Includes 'generate_response' context flag set to True.
    """
    task_id = f"task_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    return {
        "task_id": task_id,
        "conversation_id": conversation_id,
        "message": message,
        "speaker": speaker,
        "timestamp": time.time(),
        "priority": priority,
        "context": {"generate_response": True},
    }


def _log_worker_status(worker: ConversationWorker) -> None:
    """
    Logs the current status of the ConversationWorker.

    Retrieves and logs key metrics like state, queue size, processed counts,
    and recent errors.

    Args:
        worker: The ConversationWorker instance to query.
    """
    try:
        status = worker.get_status()
        logger.info(f"Worker status: {status.get('state', 'N/A')}")
        logger.info(f"Queue size: {status.get('queue_size', 'N/A')}")
        logger.info(f"Processed: {status.get('processed_count', 'N/A')}")
        logger.info(f"Success: {status.get('success_count', 'N/A')}")
        logger.info(f"Errors: {status.get('error_count', 'N/A')}")

        recent_errors = status.get("recent_errors")
        if recent_errors:
            logger.warning(f"Recent errors: {recent_errors}")
    except Exception as e:
        logger.error(f"Failed to retrieve worker status: {e}", exc_info=True)


@eidosian()
def main() -> None:
    """
    Demonstrates the setup and operation of the ConversationWorker.

    Initializes necessary components (DB, Emotion, Queue, Graph, VectorStore,
    Models, ConversationManager, ParserRefiner), creates and starts a
    ConversationWorker, submits sample conversation tasks, monitors the
    worker's status, demonstrates pause/resume functionality, and finally
    stops the worker gracefully.
    """
    worker: Optional[ConversationWorker] = None
    db_manager: Optional[DBManager] = None

    try:
        # Component Initialization
        logger.info("Initializing Word Forge components...")

        # Database Manager
        db_manager = DBManager(db_path=DB_PATH)
        db_manager.create_tables()
        logger.info(f"Using database at {DB_PATH}")

        # Emotion Manager
        emotion_manager = EmotionManager(db_manager)

        # Graph Manager
        graph_manager = GraphManager(db_manager)

        # Vector Store
        vector_store = VectorStore()

        # Conversation Models - Instantiate them
        llm_state = ModelState()
        reflexive_model_instance = ReflexiveLanguageModel(llm_state=llm_state)
        lightweight_model_instance = LightweightLanguageModel(llm_state=llm_state)
        affective_model_instance = AffectiveLexicalLanguageModel(llm_state=llm_state)
        identity_model_instance = EidosianIdentityModel(llm_state=llm_state)

        # Conversation Manager
        conversation_manager = ConversationManager(
            db_manager=db_manager,
            emotion_manager=emotion_manager,
            graph_manager=graph_manager,
            vector_store=vector_store,
            reflexive_model=reflexive_model_instance,
            lightweight_model=lightweight_model_instance,
            affective_model=affective_model_instance,
            identity_model=identity_model_instance,
        )

        # Queue Managers
        queue_manager_for_parser = QueueManager[str]()
        # Use the explicit type expected by ConversationWorker
        queue_manager_for_worker = QueueManager[ConversationTask | str | Dict[str, Any]]()

        # Parser Refiner
        parser_refiner = ParserRefiner(db_manager, queue_manager_for_parser)

        # Conversation Worker
        worker = ConversationWorker(
            parser_refiner=parser_refiner,
            queue_manager=queue_manager_for_worker,
            conversation_manager=conversation_manager,
            db_manager=db_manager,
            emotion_manager=emotion_manager,
            poll_interval=WORKER_POLL_INTERVAL,
            processing_timeout=WORKER_PROCESSING_TIMEOUT,
            batch_size=WORKER_BATCH_SIZE,
            enable_logging=True,
        )
        logger.info("Conversation worker initialized successfully.")

        # Worker Operation
        logger.info("Starting worker thread...")
        worker.start()

        # Task Submission
        logger.info("Submitting sample conversation tasks...")
        sample_conversations: List[List[Tuple[str, str]]] = [
            [
                ("user", "Tell me about algorithms"),
                ("user", "How are algorithms used in AI?"),
                ("user", "Thanks for the explanation"),
            ],
            [
                ("user", "What is machine learning?"),
                ("user", "Can you explain neural networks?"),
            ],
        ]

        for conversation_data in sample_conversations:
            result = conversation_manager.start_conversation()
            if result.is_success:
                conversation_id = result.unwrap()
                logger.info(f"Created conversation {conversation_id}")

                for speaker, message in conversation_data:
                    task = _create_conversation_task(conversation_id, speaker, message)
                    queue_manager_for_worker.enqueue(task)
                    logger.info(f"Enqueued task {task['task_id']} for conv {conversation_id}: '{message}'")
                    time.sleep(0.1)
            else:
                logger.error(f"Failed to start conversation: {result.unwrap_err()}")

        # Monitoring
        logger.info("Monitoring worker status...")
        for i in range(MONITORING_ITERATIONS):
            logger.info(f"Monitoring check {i+1}/{MONITORING_ITERATIONS}...")
            _log_worker_status(worker)
            time.sleep(MONITORING_INTERVAL)

        # Pause/Resume Demonstration
        logger.info("Demonstrating pause/resume...")
        logger.info("Pausing worker...")
        worker.pause()
        time.sleep(MONITORING_INTERVAL / 2)
        _log_worker_status(worker)

        logger.info("Resuming worker...")
        worker.resume()
        time.sleep(MONITORING_INTERVAL)
        _log_worker_status(worker)

        # Final Status Check
        logger.info("Retrieving final worker status before shutdown...")
        final_status = worker.get_status()
        logger.info("--- Final Worker Status ---")
        for key, value in final_status.items():
            if key != "conversation_metrics":
                logger.info(f"  {key}: {value}")
        logger.info("--- Conversation Metrics ---")
        conv_metrics = final_status.get("conversation_metrics", {})
        if conv_metrics:
            for key, value in conv_metrics.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("  No conversation metrics available.")
        logger.info("---------------------------")

    except Exception as e:
        logger.error(f"An unexpected error occurred in the demo: {e}", exc_info=True)

    finally:
        if worker and worker.is_alive():
            logger.info("Stopping worker...")
            worker.stop()
            worker.join(timeout=SHUTDOWN_TIMEOUT)
            if worker.is_alive():
                logger.warning("Worker thread did not terminate gracefully.")
            else:
                logger.info("Worker shutdown complete.")
        elif worker:
            logger.info("Worker was already stopped or not started.")
        else:
            logger.info("Worker instance was not created.")


if __name__ == "__main__":
    main()
