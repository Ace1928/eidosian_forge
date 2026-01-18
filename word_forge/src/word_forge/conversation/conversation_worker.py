"""
Conversation Worker Module for Word Forge.

This module provides a thread-based worker for continuous processing of
conversation tasks from a queue. It handles message generation, context tracking,
and asynchronous conversation management with comprehensive lifecycle control
and metrics tracking.

Architecture:
    ┌────────────────────┐
    │ ConversationWorker │
    └──────────┬─────────┘
               │
    ┌──────────┴──────────┐
    │     Components      │
    └─────────────────────┘
    ┌───────┬────────┬─────────┐
    │Queue  │Parser  │Conversa-│
    │Manager│Refiner │tionMgr  │
    └───────┴────────┴─────────┘
"""

from __future__ import annotations

import logging
import queue
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Event, Lock, RLock, Thread
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast, final

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager, Result  # Import Result

# Configure module logger
logger = logging.getLogger(__name__)


class ConversationWorkerState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()
    PAUSED = auto()
    RECOVERY = auto()

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class ConversationError(Exception):
    """Base exception for conversation worker errors."""

    pass


class ConversationProcessingError(ConversationError):
    """Raised when processing a conversation task fails."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.context = context or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        error_msg = str(self.args[0]) if self.args else ""
        if self.cause:
            error_msg += f" | Cause: {self.cause}"
        if self.context:
            error_msg += f" | Context: {self.context}"
        return error_msg


class ConversationTimeoutError(ConversationError):
    """Raised when a conversation task processing times out."""

    pass


class ConversationQueueError(ConversationError):
    """Raised when queue operations fail."""

    pass


class TaskResult(Enum):
    """Result status of a conversation task processing attempt."""

    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    DEFERRED = auto()
    INVALID = auto()


class ConversationWorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    processed_count: int
    success_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str
    queue_size: int
    recent_errors: List[str]
    next_poll: Optional[float]
    conversation_metrics: Dict[str, Any]
    retry_queue_size: int


class ConversationTask(TypedDict, total=False):
    """
    Type definition for a conversation task placed on the queue.

    Attributes:
        task_id: Unique identifier for this task instance.
        conversation_id: ID of the conversation this task belongs to (optional for new convos).
        message: The textual content of the message to process or respond to.
        speaker: Identifier of the message speaker.
        timestamp: Time the task was created or the message occurred.
        priority: Task priority for queue management.
        context: Dictionary containing additional processing instructions or data
                 (e.g., `generate_response: True`).
        completion_callback: Optional function to call upon task completion.
    """


@dataclass
class ProcessingMetrics:
    """Tracks and aggregates processing metrics with thread safety."""

    processed_count: int = 0
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    deferred_count: int = 0
    invalid_count: int = 0
    processing_times: List[float] = field(default_factory=list)
    task_types: Dict[str, int] = field(default_factory=lambda: {})
    error_types: Dict[str, int] = field(default_factory=lambda: {})
    _lock: Lock = field(default_factory=Lock)

    def record_processed(self) -> None:
        """Increment the processed count thread-safely."""
        self._lock.acquire()
        try:
            self.processed_count += 1
        finally:
            self._lock.release()

    def record_result(self, result: TaskResult, processing_time: float) -> None:
        """Record a processing result with timing information."""
        self._lock.acquire()
        try:
            if result == TaskResult.SUCCESS:
                self.success_count += 1
            elif result == TaskResult.FAILURE:
                self.error_count += 1
            elif result == TaskResult.TIMEOUT:
                self.timeout_count += 1
            elif result == TaskResult.DEFERRED:
                self.deferred_count += 1
            elif result == TaskResult.INVALID:
                self.invalid_count += 1

            self.processing_times.append(processing_time)
            # Keep only the last 100 processing times
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
        finally:
            self._lock.release()

    def record_task_type(self, task_type: str) -> None:
        """Record a task type for distribution analysis."""
        self._lock.acquire()
        try:
            self.task_types[task_type] = self.task_types.get(task_type, 0) + 1
        finally:
            self._lock.release()

    def record_error(self, error_type: str) -> None:
        """Record an error type for error distribution analysis."""
        self._lock.acquire()
        try:
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        finally:
            self._lock.release()

    def get_avg_processing_time(self) -> Optional[float]:
        """Calculate the average processing time with timeout protection."""
        try:
            # Use a timeout to prevent deadlocks during shutdown
            if not self._lock.acquire(timeout=0.5):
                return None

            try:
                if not self.processing_times:
                    return None
                return sum(self.processing_times) / len(self.processing_times)
            finally:
                self._lock.release()
        except Exception:
            # Safely handle any exceptions during shutdown
            return None

    def get_metrics_dict(self) -> Dict[str, Any]:
        """
        Return a dictionary of all metrics, calculated thread-safely.

        Includes counts for processed, success, error, timeout, deferred,
        and invalid tasks, average processing time, task type distribution,
        error type distribution, and overall success rate. Handles potential
        deadlocks during acquisition with timeouts.

        Returns:
            Dict[str, Any]: A dictionary containing aggregated metrics. Returns
                            minimal info with an error message if lock acquisition fails.
        """
        try:
            # Use a timeout to prevent deadlocks
            if not self._lock.acquire(timeout=1.0):
                # Return minimal metrics if we can't acquire the lock
                return {
                    "processed_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "success_rate": 0,
                    "error": "Metrics lock acquisition timed out",
                }

            try:
                # Calculate average processing time outside the lock
                avg_time = None
                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)

                return {
                    "processed_count": self.processed_count,
                    "success_count": self.success_count,
                    "error_count": self.error_count,
                    "timeout_count": self.timeout_count,
                    "deferred_count": self.deferred_count,
                    "invalid_count": self.invalid_count,
                    "avg_processing_time_ms": (
                        round(avg_time * 1000, 2) if avg_time is not None else None
                    ),
                    "task_type_distribution": dict(self.task_types),
                    "error_type_distribution": dict(self.error_types),
                    "success_rate": round(
                        self.success_count / max(1, self.processed_count) * 100, 2
                    ),
                }
            finally:
                self._lock.release()
        except Exception as e:
            # Return error metrics if something goes wrong
            return {"error": f"Failed to get metrics: {str(e)}", "processed_count": 0}


class StateTracker:
    """
    Manages worker state transitions and status reporting thread-safely.

    Tracks the current lifecycle state (RUNNING, STOPPED, PAUSED, etc.),
    uptime, last update time, next scheduled poll time, and recent errors.
    Uses an RLock to ensure atomic updates and reads.
    """

    def __init__(self) -> None:
        """Initialize state tracker with default values."""
        self._lock = RLock()
        self._state = ConversationWorkerState.STOPPED
        self._start_time: Optional[float] = None
        self._last_update: Optional[float] = None
        self._next_poll: Optional[float] = None
        self._recent_errors: List[Tuple[float, str]] = []
        self._max_recent_errors = 10

    def start(self) -> None:
        """Transition to RUNNING state."""
        with self._lock:
            self._state = ConversationWorkerState.RUNNING
            self._start_time = time.time()
            self._last_update = time.time()

    def stop(self) -> None:
        """Transition to STOPPED state."""
        with self._lock:
            self._state = ConversationWorkerState.STOPPED
            self._last_update = time.time()

    def pause(self) -> None:
        """Transition to PAUSED state."""
        with self._lock:
            self._state = ConversationWorkerState.PAUSED
            self._last_update = time.time()

    def resume(self) -> None:
        """Transition to RUNNING state from PAUSED."""
        with self._lock:
            if self._state == ConversationWorkerState.PAUSED:
                self._state = ConversationWorkerState.RUNNING
                self._last_update = time.time()

    def error(self) -> None:
        """Transition to ERROR state."""
        with self._lock:
            self._state = ConversationWorkerState.ERROR
            self._last_update = time.time()

    def recovery(self) -> None:
        """Transition to RECOVERY state."""
        with self._lock:
            self._state = ConversationWorkerState.RECOVERY
            self._last_update = time.time()

    def record_update(self) -> None:
        """Record a successful update to the state."""
        with self._lock:
            self._last_update = time.time()

    def record_error(self, error_msg: str) -> None:
        """Record an error with timestamp."""
        with self._lock:
            self._recent_errors.append((time.time(), error_msg))
            # Keep only the most recent errors
            if len(self._recent_errors) > self._max_recent_errors:
                self._recent_errors = self._recent_errors[-self._max_recent_errors :]

    def set_next_poll(self, next_poll: float) -> None:
        """Set the next scheduled poll time."""
        with self._lock:
            self._next_poll = next_poll

    @property
    def state(self) -> ConversationWorkerState:
        """Get the current state."""
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Check if the worker is in RUNNING state."""
        with self._lock:
            return self._state == ConversationWorkerState.RUNNING

    @property
    def uptime(self) -> Optional[float]:
        """Calculate uptime in seconds if started."""
        with self._lock:
            if self._start_time is None:
                return None
            return time.time() - self._start_time

    @property
    def last_update(self) -> Optional[float]:
        """Get timestamp of last update."""
        with self._lock:
            return self._last_update

    @property
    def next_poll(self) -> Optional[float]:
        """Get the next scheduled poll time."""
        with self._lock:
            return self._next_poll

    @property
    def recent_errors(self) -> List[str]:
        """Get list of recent error messages."""
        with self._lock:
            return [
                error
                for _, error in sorted(
                    self._recent_errors, key=lambda x: x[0], reverse=True
                )
            ]

    def to_dict(
        self, queue_size: int, retry_queue_size: int, metrics: Dict[str, Any]
    ) -> ConversationWorkerStatus:
        """
        Convert current state and metrics into a status dictionary.

        Args:
            queue_size: Current size of the main processing queue.
            retry_queue_size: Current size of the internal retry queue.
            metrics: Dictionary of performance metrics from ProcessingMetrics.

        Returns:
            ConversationWorkerStatus: A dictionary summarizing the worker's current state.
        """
        with self._lock:
            uptime_val = self.uptime
            return ConversationWorkerStatus(
                running=self.is_running,
                state=str(self._state),
                processed_count=metrics.get(
                    "processed_count", 0
                ),  # Use .get for safety
                success_count=metrics.get("success_count", 0),
                error_count=metrics.get("error_count", 0),
                last_update=self._last_update,
                uptime=(
                    round(uptime_val, 2) if uptime_val is not None else None
                ),  # Round uptime
                queue_size=queue_size,
                retry_queue_size=retry_queue_size,  # Include retry queue size
                recent_errors=self.recent_errors[:5],  # Limit recent errors shown
                next_poll=self._next_poll,
                conversation_metrics=metrics,
            )


@final
class ConversationWorker(Thread):
    """
    Asynchronous worker thread for processing conversation tasks.

    Continuously monitors a queue for incoming `ConversationTask` items.
    Processes tasks by interacting with the `ConversationManager` to add
    messages and trigger the multi-model response generation pipeline.
    Handles lifecycle management (start, stop, pause, resume), error recovery
    with retries and backoff, and detailed metrics tracking.

    Designed for robust, continuous operation within the Word Forge system.
    """

    def __init__(
        self,
        parser_refiner: ParserRefiner,
        queue_manager: QueueManager[Union[ConversationTask, str, Dict[str, Any]]],
        conversation_manager: ConversationManager,
        db_manager: DBManager,
        emotion_manager: Optional[EmotionManager] = None,
        poll_interval: float = 2.0,
        processing_timeout: float = 30.0,
        batch_size: int = 1,
        max_retries: int = 3,
        daemon: bool = True,
        enable_logging: bool = True,
    ) -> None:
        """
        Initialize conversation worker with required components.

        Args:
            parser_refiner: Parser component for lexical processing
            queue_manager: Queue system for message retrieval
            conversation_manager: Manager for conversation persistence
            db_manager: Database access for storage
            emotion_manager: Optional emotion analysis component
            poll_interval: Seconds between queue polling attempts
            processing_timeout: Maximum seconds to process a task
            batch_size: Number of tasks to process per cycle
            max_retries: Maximum retry attempts for failed tasks
            daemon: Whether the thread should be a daemon
            enable_logging: Whether to log processing activity
        """
        super().__init__(name="ConversationWorker", daemon=daemon)
        self.parser_refiner = parser_refiner
        self.queue_manager = queue_manager
        self.conversation_manager = conversation_manager
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager

        # Configuration
        self.poll_interval = poll_interval
        self.processing_timeout = processing_timeout
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.enable_logging = enable_logging

        # State management
        self._stop_event = Event()
        self._pause_event = Event()
        self.metrics = ProcessingMetrics()
        self.state_tracker = StateTracker()

        # Advanced tracking
        self._retry_queue: Dict[str, Tuple[ConversationTask, int]] = {}
        self._processing_times: Dict[str, float] = {}
        self._conversation_cache: Dict[int, Dict[str, Any]] = {}

        # Logger setup
        self.logger = logger if enable_logging else None
        if self.logger:
            self.logger.info("ConversationWorker initialized")

    def start(self) -> None:
        """
        Start the conversation worker thread.

        If the worker is already running, this method has no effect.
        """
        if self.is_alive():
            if self.logger:
                self.logger.warning("Worker thread already started")
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self.state_tracker.start()

        if self.logger:
            self.logger.info("Starting conversation worker thread")

        super().start()

    def stop(self) -> None:
        """
        Signal the worker thread to stop processing and shut down.

        This method returns immediately, but the thread may continue
        until the current task is completed.
        """
        if not self.is_alive():
            return

        if self.logger:
            self.logger.info("Stopping conversation worker")

        self._stop_event.set()
        self._pause_event.clear()  # Ensure thread isn't paused when stopping
        self.state_tracker.stop()

    def restart(self) -> None:
        """
        Restart the worker thread by stopping it and starting a new one.

        If the thread is not alive, it will just be started.
        """
        if self.is_alive():
            self.stop()
            # Wait for thread to actually terminate
            self.join(timeout=min(self.processing_timeout, 10.0))

        # Reset internal state
        self._stop_event.clear()
        self._pause_event.clear()
        self._retry_queue.clear()
        self._processing_times.clear()

        # Start a new thread
        self.state_tracker.start()
        super().start()

        if self.logger:
            self.logger.info("Conversation worker restarted")

    def pause(self) -> None:
        """
        Pause the worker thread temporarily.

        The thread will finish processing the current task and then wait
        until resumed or stopped.
        """
        if not self.is_alive() or self._stop_event.is_set():
            return

        self._pause_event.set()
        self.state_tracker.pause()

        if self.logger:
            self.logger.info("Conversation worker paused")

    def resume(self) -> None:
        """
        Resume a paused worker thread.

        If the thread is not paused, this method has no effect.
        """
        if not self.is_alive() or self._stop_event.is_set():
            return

        self._pause_event.clear()
        self.state_tracker.resume()

        if self.logger:
            self.logger.info("Conversation worker resumed")

    def run(self) -> None:
        """
        Main execution loop for the worker thread.

        Continuously polls the queue, processes tasks in batches, handles
        retries, manages pause/stop states, implements error recovery with
        exponential backoff, and logs activity. Terminates when the stop
        event is set.
        """
        if self.logger:
            self.logger.info("Conversation worker thread started")

        last_error_recovery = 0.0
        backoff_factor = 1.0

        try:
            while not self._stop_event.is_set():
                # Handle pause state
                if self._pause_event.is_set():
                    time.sleep(0.5)  # Lighter sleep during pause
                    continue

                # Set next poll time for status reporting
                next_poll = time.time() + self.poll_interval
                self.state_tracker.set_next_poll(next_poll)

                try:
                    # Process retry queue first
                    self._process_retry_queue()

                    # Process new tasks from queue
                    processed = self._process_queue_batch()

                    if processed > 0:
                        # Reset backoff on successful processing
                        backoff_factor = 1.0
                        self.state_tracker.record_update()
                    else:
                        # Wait before next poll
                        time.sleep(self.poll_interval * backoff_factor)

                except Exception as e:
                    error_name = type(e).__name__
                    error_msg = f"{error_name}: {str(e)}"
                    self.state_tracker.record_error(error_msg)
                    self.metrics.record_error(error_name)

                    if self.logger:
                        self.logger.error(
                            f"Error in conversation worker: {error_msg}", exc_info=True
                        )

                    # Handle recovery
                    current_time = time.time()
                    if (
                        current_time - last_error_recovery > 60.0
                    ):  # 1 minute between recoveries
                        self.state_tracker.recovery()
                        if self.logger:
                            self.logger.info("Entering recovery mode")

                        # Recovery logic
                        self._clear_stale_processing()
                        last_error_recovery = current_time
                        self.state_tracker.resume()  # Back to running after recovery

                    # Exponential backoff capped at 60 seconds
                    backoff_factor = min(backoff_factor * 2, 60.0 / self.poll_interval)
                    time.sleep(self.poll_interval * backoff_factor)

        except Exception as e:
            if self.logger:
                self.logger.critical(
                    f"Critical error in conversation worker: {str(e)}", exc_info=True
                )
            self.state_tracker.error()

        finally:
            if self.logger:
                self.logger.info("Conversation worker thread terminated")

    def _process_retry_queue(self) -> int:
        """
        Attempts to re-process tasks currently in the retry queue.

        Iterates through tasks marked for retry. If a task succeeds, it's
        removed. If it fails again, its retry count is incremented. Tasks
        exceeding `max_retries` are discarded.

        Returns:
            int: The number of retry tasks attempted in this cycle.
        """
        if not self._retry_queue:
            return 0

        processed = 0
        retry_items = list(self._retry_queue.items())

        for task_id, (task, attempts) in retry_items:
            # Skip if we've reached max retries
            if attempts >= self.max_retries:
                if self.logger:
                    self.logger.warning(
                        f"Task {task_id} exceeded maximum retry attempts ({self.max_retries})"
                    )
                del self._retry_queue[task_id]
                continue

            # Process the retry
            try:
                result = self._process_task(task)
                processed += 1

                if result == TaskResult.SUCCESS:
                    # Task succeeded, remove from retry queue
                    del self._retry_queue[task_id]
                elif result == TaskResult.DEFERRED:
                    # Keep in retry queue but don't increment attempts
                    pass
                else:
                    # Increment retry attempt count
                    self._retry_queue[task_id] = (task, attempts + 1)

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error processing retry task {task_id}: {str(e)}",
                        exc_info=True,
                    )
                # Increment retry attempt count
                self._retry_queue[task_id] = (task, attempts + 1)

        return processed

    def _process_queue_batch(self) -> int:
        """
        Dequeues and processes a batch of tasks from the main queue.

        Attempts to dequeue up to `batch_size` tasks. Handles different
        task item types (ConversationTask dicts, simple strings) and logs
        errors during dequeuing or processing.

        Returns:
            int: The number of tasks successfully dequeued and initiated for processing.
        """
        processed_count = 0

        for _ in range(self.batch_size):
            try:
                # Get a task result from the queue
                dequeue_result: Result[Union[ConversationTask, str, Dict[str, Any]]] = (
                    self.queue_manager.dequeue(block=False)
                )

                if dequeue_result.is_success:
                    task_item = dequeue_result.unwrap()

                    # Handle different types of queue items
                    if isinstance(task_item, dict) and "task_id" in task_item:
                        # This is a valid ConversationTask
                        task = cast(ConversationTask, task_item)
                        self._process_task(task)
                        processed_count += 1
                    elif isinstance(task_item, str):
                        # Handle string tasks by converting to simple task format
                        simple_task = cast(
                            ConversationTask,
                            {
                                "task_id": f"auto_{int(time.time())}",
                                "conversation_id": None,
                                "message": task_item,
                                "speaker": "user",
                                "timestamp": time.time(),
                                "priority": 1,
                                "context": {},
                            },
                        )
                        self._process_task(simple_task)
                        processed_count += 1
                    else:
                        # Skip invalid task types with more informative logging
                        task_type = type(task_item).__name__
                        if self.logger:
                            self.logger.warning(
                                f"Invalid task type received: {task_type}. Expected ConversationTask. "
                                f"Item: {str(task_item)[:100]}... (truncated)"
                            )
                else:
                    # Handle error from dequeue_result
                    error = dequeue_result.error
                    if error is not None and error.error_code != "EMPTY_QUEUE":
                        if self.logger:
                            self.logger.error(f"Error dequeuing task: {error.message}")
                        self.state_tracker.record_error(
                            f"Dequeue error: {error.message}"
                        )

            except queue.Empty:
                # No more tasks in queue
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error processing queue batch: {e}", exc_info=True
                    )
                self.state_tracker.record_error(str(e))

        return processed_count

    def _process_task(self, task: ConversationTask) -> TaskResult:
        """
        Processes a single conversation task.

        Validates the task, interacts with the `ConversationManager` to add
        the message, potentially triggers response generation via the manager,
        and executes any completion callback. Records metrics and handles errors.

        Args:
            task: The `ConversationTask` dictionary to process.

        Returns:
            TaskResult: Enum indicating the outcome (SUCCESS, FAILURE, INVALID, etc.).

        Raises:
            ConversationProcessingError: If an unrecoverable error occurs during processing.
                                         The original exception is attached as the cause.
        """
        task_id = task.get("task_id", str(id(task)))
        start_time = time.time()

        # Record task being processed
        self.metrics.record_processed()
        self._processing_times[task_id] = start_time

        try:
            # Validate task data
            message = task.get("message", "")
            if not message:
                if self.logger:
                    self.logger.warning(f"Task {task_id} has empty message")
                self.metrics.record_result(TaskResult.INVALID, time.time() - start_time)
                return TaskResult.INVALID

            # Extract task parameters
            conversation_id = task.get("conversation_id")
            speaker = task.get("speaker", "user")
            context = task.get("context", {})

            # Process based on task type (determine by context and existence of conversation_id)
            task_type = "add_message"  # Default task type
            if conversation_id is None and message:
                task_type = "start_conversation_and_add"
            elif context.get("generate_response", False):
                task_type = "add_message_and_respond"

            self.metrics.record_task_type(task_type)  # Record task type

            if task_type == "start_conversation_and_add":
                # Create new conversation
                start_result = self.conversation_manager.start_conversation()
                if not start_result.is_success:
                    raise ConversationProcessingError(
                        f"Task {task_id}: Failed to start new conversation: {start_result.error}"
                    )
                conversation_id = start_result.unwrap()
                if self.logger:
                    self.logger.info(
                        f"Task {task_id}: Started new conversation {conversation_id}"
                    )

            # Ensure conversation_id is valid before proceeding
            if conversation_id is None:
                raise ConversationProcessingError(
                    f"Task {task_id}: Missing conversation ID for processing."
                )

            # Add message to conversation - response generation is handled by the manager now
            # if generate_response is True in the context.
            message_id = self.conversation_manager.add_message(
                conversation_id,
                speaker,
                message,
                generate_response=context.get("generate_response", False),
            )

            if self.logger:
                self.logger.debug(
                    f"Task {task_id}: Added message {message_id} to conversation {conversation_id}. Response generation handled by manager."
                )

            # Response generation is now implicitly handled within add_message if requested.
            # No need for explicit _generate_response call here.

            # Execute callback if provided
            completion_callback = task.get("completion_callback")
            if completion_callback and callable(completion_callback):
                result_data: Dict[str, Any] = {
                    "task_id": task_id,
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "processed_at": time.time(),
                    "success": True,
                }
                completion_callback(result_data)

            # Record successful processing
            processing_time = time.time() - start_time
            self.metrics.record_result(TaskResult.SUCCESS, processing_time)

            # Clean up processing tracking
            if task_id in self._processing_times:
                del self._processing_times[task_id]

            return TaskResult.SUCCESS

        except Exception as e:
            # Record error
            error_type = type(e).__name__
            processing_time = time.time() - start_time
            self.metrics.record_error(error_type)
            self.metrics.record_result(TaskResult.FAILURE, processing_time)

            # Log error
            if self.logger:
                self.logger.error(
                    f"Error processing task {task_id}: {str(e)}", exc_info=True
                )

            # Clean up processing tracking
            if task_id in self._processing_times:
                del self._processing_times[task_id]

            # Re-raise as conversation error
            context: Dict[str, Any] = {
                "task_id": task_id,
                "conversation_id": task.get("conversation_id"),
                "processing_time": processing_time,
            }
            raise ConversationProcessingError(
                f"Failed to process conversation task: {str(e)}",
                cause=e,
                context=context,
            )

    def _clear_stale_processing(self) -> None:
        """
        Identifies and removes tracking entries for tasks that have exceeded the timeout.

        Iterates through the `_processing_times` dictionary and removes entries
        older than `processing_timeout` seconds, logging a warning for each.
        """
        current_time = time.time()
        stale_tasks: List[str] = []

        for task_id, start_time in self._processing_times.items():
            if current_time - start_time > self.processing_timeout:
                stale_tasks.append(task_id)

        for task_id in stale_tasks:
            if self.logger:
                self.logger.warning(f"Clearing stale task {task_id}")
            del self._processing_times[task_id]

    def get_status(self) -> ConversationWorkerStatus:
        """
        Retrieves the current operational status and metrics of the worker.

        Combines state information from `StateTracker` with metrics from
        `ProcessingMetrics` and current queue sizes.

        Returns:
            ConversationWorkerStatus: A dictionary summarizing the worker's status.
        """
        queue_size = self.queue_manager.size
        retry_queue_size = len(self._retry_queue)
        metrics_dict = self.metrics.get_metrics_dict()

        return self.state_tracker.to_dict(queue_size, retry_queue_size, metrics_dict)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retrieves the current performance metrics collected by the worker.

        Returns the dictionary generated by `ProcessingMetrics.get_metrics_dict()`.

        Returns:
            Dict[str, Any]: A dictionary containing detailed performance metrics.
        """
        return self.metrics.get_metrics_dict()

    def process_task(self, task: ConversationTask) -> bool:
        """
        Processes a single task synchronously, bypassing the queue.

        Useful for immediate execution needs or testing. Directly calls the
        internal `_process_task` method.

        Args:
            task: The `ConversationTask` to process.

        Returns:
            bool: True if the task completed successfully (TaskResult.SUCCESS),
                  False otherwise.
        """
        try:
            result = self._process_task(task)
            return result == TaskResult.SUCCESS
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error in synchronous task processing: {str(e)}", exc_info=True
                )
            return False

    def submit_message(
        self, conversation_id: Optional[int], message: str, speaker: str
    ) -> Optional[str]:
        """
        Creates and enqueues a standard conversation task to add a message.

        Sets the task context to trigger response generation by default.

        Args:
            conversation_id: The target conversation ID, or None to start a new one.
            message: The message text.
            speaker: The speaker identifier.

        Returns:
            Optional[str]: The generated unique task ID if enqueuing was successful,
                           otherwise None.
        """
        try:
            task_id = f"task_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

            task = cast(
                ConversationTask,
                {
                    "task_id": task_id,
                    "conversation_id": conversation_id,
                    "message": message,
                    "speaker": speaker,
                    "timestamp": time.time(),
                    "priority": 1,
                    "context": {"generate_response": True},
                },
            )

            # Enqueue the task
            self.queue_manager.enqueue(task)

            return task_id

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error submitting message: {str(e)}", exc_info=True)
            return None
