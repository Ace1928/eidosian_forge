import logging
import random
import sqlite3
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, auto
from types import TracebackType
from typing import (
    Any,
    DefaultDict,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypedDict,
    final,
)

from eidosian_core import eidosian
from word_forge.config import config
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.emotion.emotion_processor import RecursiveEmotionProcessor
from word_forge.emotion.emotion_types import EmotionDimension

# Conditionally import recursive processor


RECURSIVE_PROCESSOR_AVAILABLE = True


class EmotionState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()
    PAUSED = auto()  # New state for graceful pause
    RECOVERY = auto()  # New state for automatic recovery

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class EmotionError(Exception):
    """Base exception for emotion processing errors."""

    pass


class EmotionDBError(EmotionError):
    """Raised when database operations for emotion processing fail."""

    pass


class EmotionProcessingError(EmotionError):
    """Raised when word emotion classification fails."""

    pass


@dataclass(frozen=True)
class WordEmotion:
    """Immutable data structure representing a word needing emotional classification."""

    word_id: int
    term: str


class EmotionWorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    processed_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str
    recent_errors: List[str]  # Added field for recent error types
    backlog_estimate: int  # Added field for estimated backlog
    strategy: str  # Added field for current strategy
    next_poll: Optional[float]  # Added field for next poll time


class EmotionWorkerInterface(Protocol):
    """Protocol defining the required interface for emotion workers."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...  # Added restart method
    def pause(self) -> None: ...  # Added pause method
    def resume(self) -> None: ...  # Added resume method
    def get_status(self) -> EmotionWorkerStatus: ...
    def get_metrics(self) -> Dict[str, Any]: ...  # Added get_metrics
    def is_alive(self) -> bool: ...


class EmotionAssignmentStrategy(Enum):
    """Strategy for assigning emotional values to words."""

    RANDOM = "random"  # Pure random assignment within ranges
    RECURSIVE = "recursive"  # Using recursive emotion processor
    HYBRID = "hybrid"  # Combine multiple strategies

    @classmethod
    def from_string(cls, value: str) -> "EmotionAssignmentStrategy":
        """Convert string to enum value with fallback to RANDOM."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.RANDOM


class StateTracker:
    """Encapsulates worker state tracking with thread-safe operations."""

    def __init__(self) -> None:
        """Initialize the state tracker with default values."""
        self._lock = threading.RLock()
        self._state = EmotionState.STOPPED
        self._start_time: Optional[float] = None
        self._last_update: Optional[float] = None
        self._next_poll: Optional[float] = None
        self.processed_count = 0
        self._error_count = 0
        self._recent_errors: Deque[str] = deque(maxlen=5)
        self._backlog_estimate = 0
        self._strategy = EmotionAssignmentStrategy.RANDOM

    @eidosian()
    def to_dict(self, is_alive: bool, stop_flag: bool) -> EmotionWorkerStatus:
        """Convert current state to a status dictionary."""
        with self._lock:
            uptime = None
            if self._start_time:
                uptime = time.time() - self._start_time

            return {
                "running": is_alive and not stop_flag,
                "processed_count": self.processed_count,
                "error_count": self._error_count,
                "last_update": self._last_update,
                "next_poll": self._next_poll,
                "uptime": uptime,
                "state": str(self._state),
                "recent_errors": list(self._recent_errors),
                "backlog_estimate": self._backlog_estimate,
                "strategy": self._strategy.value,
            }

    @eidosian()
    def set_strategy(self, strategy: EmotionAssignmentStrategy) -> None:
        """Update the current processing strategy."""
        with self._lock:
            self._strategy = strategy

    @eidosian()
    def start(self) -> None:
        """Mark the worker as started."""
        with self._lock:
            self._start_time = time.time()
            self._state = EmotionState.RUNNING

    @eidosian()
    def stop(self) -> None:
        """Mark the worker as stopped."""
        with self._lock:
            self._state = EmotionState.STOPPED

    @eidosian()
    def pause(self) -> None:
        """Mark the worker as paused."""
        with self._lock:
            self._state = EmotionState.PAUSED

    @eidosian()
    def resume(self) -> None:
        """Resume the worker from paused state."""
        with self._lock:
            self._state = EmotionState.RUNNING

    @eidosian()
    def increment_processed(self) -> None:
        """Record a successful word processing."""
        with self._lock:
            self.processed_count += 1
            self._last_update = time.time()

    @eidosian()
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._error_count += 1
            self._recent_errors.append(error_type)
            self._state = EmotionState.ERROR

    @eidosian()
    def record_recovery(self) -> None:
        """Record recovery from error state."""
        with self._lock:
            self._state = EmotionState.RUNNING

    @eidosian()
    def update_backlog(self, count: int) -> None:
        """Update the estimated backlog of words to process."""
        with self._lock:
            self._backlog_estimate = count

    @eidosian()
    def set_next_poll(self, timestamp: float) -> None:
        """Set the timestamp for the next polling operation."""
        with self._lock:
            self._next_poll = timestamp

    @property
    def error_count(self) -> int:
        """Get the current error count."""
        with self._lock:
            return self._error_count

    @property
    def state(self) -> EmotionState:
        """Get the current worker state."""
        with self._lock:
            return self._state

    @property
    def last_update(self) -> Optional[float]:
        """Timestamp of the last successful processing operation."""
        with self._lock:
            return self._last_update


class ErrorTracker:
    """Tracks and categorizes errors with exponential backoff logic."""

    def __init__(self, max_errors: int = 10, base_backoff: float = 1.0) -> None:
        """Initialize error tracker with backoff parameters."""
        self._lock = threading.RLock()
        self._error_counts: DefaultDict[str, int] = defaultdict(int)
        self._consecutive_errors = 0
        self._max_errors = max_errors
        self._base_backoff = base_backoff
        self._last_error_time = 0.0

    @eidosian()
    def record_error(self, error: Exception) -> float:
        """
        Record an error and calculate backoff time.

        Args:
            error: The exception that occurred

        Returns:
            The calculated backoff time in seconds
        """
        error_type = type(error).__name__

        with self._lock:
            self._error_counts[error_type] += 1
            self._consecutive_errors += 1
            self._last_error_time = time.time()

            # Calculate exponential backoff with a cap
            backoff = min(
                60.0, self._base_backoff * (2 ** min(10, self._consecutive_errors - 1))
            )
            return backoff

    @eidosian()
    def record_success(self) -> None:
        """Record a successful operation, resetting consecutive error count."""
        with self._lock:
            self._consecutive_errors = 0

    @eidosian()
    def get_error_summary(self) -> Dict[str, int]:
        """Get a summary of error counts by type."""
        with self._lock:
            return dict(self._error_counts)

    @eidosian()
    def get_recent_error_types(self, limit: int = 3) -> List[str]:
        """Get the most frequent recent error types."""
        with self._lock:
            sorted_errors = sorted(
                self._error_counts.items(), key=lambda x: x[1], reverse=True
            )
            return [error_type for error_type, _ in sorted_errors[:limit]]

    @property
    def needs_recovery(self) -> bool:
        """Check if error rate indicates need for recovery mode."""
        with self._lock:
            return self._consecutive_errors >= self._max_errors


@final
class EmotionWorker(threading.Thread):
    """
    Asynchronously assigns emotional values to unprocessed words in the database.

    This worker continuously polls the database for words without emotional
    attributes, then applies valence and arousal values to these words using
    the EmotionManager and optional RecursiveEmotionProcessor for enhanced
    emotional intelligence.

    The worker implements a multi-phase architecture:
    1. State tracking for thread-safe status monitoring
    2. Multiple emotion assignment strategies (random, recursive, hybrid)
    3. Robust error handling with exponential backoff
    4. Automatic recovery mechanisms for self-healing
    5. Configurable logging and observability
    """

    def __init__(
        self,
        db: DBManager,
        emotion_manager: EmotionManager,
        processor: Optional[Any] = None,  # Any to avoid hard dependency
        poll_interval: Optional[float] = None,
        batch_size: Optional[int] = None,
        strategy: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        daemon: bool = True,
        enable_logging: bool = True,
    ):
        """
        Initialize the emotion processing worker thread.

        Args:
            db: Database manager for accessing word data
            emotion_manager: Manager for storing emotional attributes
            processor: Optional RecursiveEmotionProcessor for advanced analysis
            poll_interval: Seconds to wait between processing cycles (defaults to config)
            batch_size: Maximum number of words to process per cycle (defaults to config)
            strategy: Emotion assignment strategy (random, recursive, hybrid)
            confidence_threshold: Minimum confidence for storing emotions
            daemon: Whether thread should run as daemon
            enable_logging: Whether to use structured logging
        """
        super().__init__(daemon=daemon)

        # Core dependencies
        self.db = db
        self.emotion_manager = emotion_manager

        # Configuration
        self.poll_interval = poll_interval or getattr(
            config.emotion, "poll_interval", 5.0
        )
        self.batch_size = batch_size or getattr(config.queue, "batch_size", 20)
        self.confidence_threshold = confidence_threshold or getattr(
            config.emotion, "min_confidence", 0.0
        )

        # Set up logging
        self.enable_logging = enable_logging
        self.logger = logging.getLogger("EmotionWorker")
        if not self.logger.handlers and enable_logging:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(config.emotion, "log_level", logging.INFO))

        # Thread control
        self._stop_flag = False
        self._pause_flag = False

        # State management
        self._state = StateTracker()
        self._errors = ErrorTracker()

        # Process strategy configuration
        strategy_str = strategy or getattr(config.emotion, "strategy", "random")
        self._strategy = EmotionAssignmentStrategy.from_string(strategy_str)
        self._state.set_strategy(self._strategy)

        # Recursive processor integration (if available)
        self._processor = None
        if processor is not None:
            self._processor = processor
            if self._strategy == EmotionAssignmentStrategy.RANDOM:
                self._strategy = EmotionAssignmentStrategy.HYBRID
                self._state.set_strategy(self._strategy)
            self.log_info(
                f"Using provided recursive processor with strategy: {self._strategy.value}"
            )
        elif RECURSIVE_PROCESSOR_AVAILABLE and self._strategy in (
            EmotionAssignmentStrategy.RECURSIVE,
            EmotionAssignmentStrategy.HYBRID,
        ):
            try:
                self._processor = RecursiveEmotionProcessor(db, emotion_manager)
                self.log_info(
                    f"Created recursive processor with strategy: {self._strategy.value}"
                )
            except Exception as e:
                self.log_error(f"Failed to initialize recursive processor: {e}")
                # Fall back to random if processor initialization fails
                if self._strategy == EmotionAssignmentStrategy.RECURSIVE:
                    self._strategy = EmotionAssignmentStrategy.RANDOM
                    self._state.set_strategy(self._strategy)
                    self.log_warning("Falling back to random strategy")

        # Metrics tracking
        self._process_times: List[float] = []
        self._observed_backlog: List[int] = []

        # Debug meta-emotion logging
        self._debug_meta_emotions = getattr(
            config.emotion, "debug_meta_emotions", False
        )

    @eidosian()
    def run(self) -> None:
        """
        Main worker loop that processes words without emotional attributes.

        Continuously polls database for unprocessed words, assigns them
        emotional values, and handles any exceptions during processing with
        intelligent error handling and recovery mechanisms.
        """
        with self._update_state("start"):
            self.log_info("Worker started")

        consecutive_success = 0
        recovery_threshold = 3  # Number of consecutive successes to exit recovery mode

        while not self._stop_flag:
            try:
                # Handle pause state if activated
                if self._pause_flag:
                    with self._update_state("pause"):
                        self.log_info("Worker paused")

                    # Sleep while paused
                    while self._pause_flag and not self._stop_flag:
                        time.sleep(0.5)

                    # If we're still not stopping, log resumed state
                    if not self._stop_flag:
                        with self._update_state("resume"):
                            self.log_info("Worker resumed")
                    continue

                # Calculate next poll time for status reporting
                next_poll = time.time() + self.poll_interval
                self._state.set_next_poll(next_poll)

                # Get unprocessed words
                start_time = time.time()
                words_to_tag = self._get_unemotioned_words()
                query_time = time.time() - start_time

                # Update backlog estimate
                if words_to_tag:
                    self._state.update_backlog(len(words_to_tag))
                    self._observed_backlog.append(len(words_to_tag))
                    self.log_info(
                        f"Processing {len(words_to_tag)} words (query: {query_time:.2f}s)"
                    )

                    # Process the words with timing
                    process_start = time.time()
                    self._process_word_emotions(words_to_tag)
                    process_time = time.time() - process_start
                    self._process_times.append(process_time)

                    # Record successful processing
                    self._errors.record_success()
                    consecutive_success += 1

                    # If we were in recovery mode and have enough success, exit recovery
                    if (
                        self._state.state == EmotionState.RECOVERY
                        and consecutive_success >= recovery_threshold
                    ):
                        with self._update_state("recovery"):
                            self.log_info(
                                "Exiting recovery mode after consistent success"
                            )
                else:
                    self.log_info("No words to process, waiting...")

                # Sleep until next poll
                time.sleep(self.poll_interval)

            except Exception as e:
                # Record the error and get backoff time
                backoff = self._errors.record_error(e)
                consecutive_success = 0

                # Log the error with context
                self._handle_processing_error(e)

                # If we hit error threshold, enter recovery mode
                if (
                    self._errors.needs_recovery
                    and self._state.state != EmotionState.RECOVERY
                ):
                    with self._update_state("recovery"):
                        self.log_warning(
                            f"Entering recovery mode due to repeated errors: {self._errors.get_recent_error_types()}"
                        )

                # Use exponential backoff for sleep time
                self.log_info(f"Backing off for {backoff:.1f}s before retrying")
                time.sleep(backoff)

        with self._update_state("stop"):
            self.log_info("Worker stopped")

    @eidosian()
    def stop(self) -> None:
        """Signal the worker thread to stop gracefully."""
        self._stop_flag = True
        self.log_info("Stop signal received")

    @eidosian()
    def pause(self) -> None:
        """
        Pause the worker without stopping the thread.

        When paused, the worker will not process any words but will remain
        responsive to status queries and other control signals.
        """
        if not self._pause_flag:
            self._pause_flag = True
            self.log_info("Pause signal received")

    @eidosian()
    def resume(self) -> None:
        """Resume a paused worker."""
        if self._pause_flag:
            self._pause_flag = False
            self.log_info("Resume signal received")

    @eidosian()
    def restart(self) -> None:
        """
        Restart the worker thread gracefully.

        Stops the current thread if running, waits for it to terminate,
        resets internal state, and starts a new thread.
        """
        self.log_info("Restart signal received")

        # Stop the current thread
        was_running = not self._stop_flag
        self.stop()

        # Wait for thread to terminate with timeout
        if self.is_alive():
            self.log_info("Waiting for worker thread to terminate...")
            self.join(timeout=5.0)

            # If thread is still alive after timeout, warn but continue
            if self.is_alive():
                self.log_warning(
                    "Worker thread did not terminate cleanly, forcing restart"
                )

        # Reset internal state
        self._stop_flag = False
        self._pause_flag = False
        self._state = StateTracker()
        self._state.set_strategy(self._strategy)
        self._errors = ErrorTracker()

        # Only restart if it was running before
        if was_running:
            self.start()
            self.log_info("Worker restarted")

    @eidosian()
    def get_status(self) -> EmotionWorkerStatus:
        """
        Return the current status of the emotion worker.

        Returns:
            Dictionary containing operational metrics including:
            - running: Whether the worker is active
            - processed_count: Number of successfully processed words
            - error_count: Number of encountered errors
            - last_update: Timestamp of last successful update
            - uptime: Seconds since thread start if running
            - state: Current worker state as string
            - recent_errors: List of most frequent error types
            - backlog_estimate: Estimated number of words waiting to be processed
            - strategy: Current emotion assignment strategy
            - next_poll: Timestamp of next scheduled database poll
        """
        return self._state.to_dict(self.is_alive(), self._stop_flag)

    @eidosian()
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed worker performance metrics.

        Returns:
            Dictionary containing comprehensive metrics and statistics
        """
        # Reuse existing performance stats method
        return self.get_performance_stats()

    @eidosian()
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics for monitoring.

        Returns:
            Dictionary with performance metrics including:
            - avg_process_time: Average time to process a batch
            - avg_batch_size: Average batch size
            - estimated_completion: Estimated time to process backlog
            - error_summary: Counts of errors by type
        """
        stats: Dict[str, Any] = {
            "error_summary": self._errors.get_error_summary(),
        }

        # Calculate average processing times if we have data
        if self._process_times:
            avg_time = sum(self._process_times) / len(self._process_times)
            stats["avg_process_time"] = avg_time

            # Calculate words per second
            if self._observed_backlog:
                avg_batch = sum(self._observed_backlog) / len(self._observed_backlog)
                stats["avg_batch_size"] = avg_batch
                words_per_second = avg_batch / avg_time if avg_time > 0 else 0
                stats["words_per_second"] = words_per_second

                # Estimate completion time if we have a backlog
                current_backlog = self._state.to_dict(self.is_alive(), self._stop_flag)[
                    "backlog_estimate"
                ]
                if current_backlog > 0 and words_per_second > 0:
                    stats["estimated_completion"] = current_backlog / words_per_second

        return stats

    def _get_unemotioned_words(self) -> List[WordEmotion]:
        """
        Retrieve words from the database that lack emotional attributes.

        Returns:
            List of WordEmotion objects representing unprocessed words

        Raises:
            EmotionDBError: If database access fails
        """
        query = """
        SELECT w.id, w.term
        FROM words w
        LEFT JOIN word_emotion we ON w.id = we.word_id
        WHERE we.word_id IS NULL
        LIMIT ?
        """
        connection = None
        try:
            connection = sqlite3.connect(self.db.db_path)
            cursor = connection.cursor()
            cursor.execute(query, (self.batch_size,))
            rows = cursor.fetchall()
            return [WordEmotion(word_id=row[0], term=row[1]) for row in rows]
        except sqlite3.Error as e:
            raise EmotionDBError(f"Failed to retrieve words: {str(e)}") from e
        finally:
            if connection is not None:
                connection.close()

    def _process_word_emotions(self, words: List[WordEmotion]) -> None:
        """
        Assign emotional values to each word and store in the database.

        Uses the selected strategy to determine emotional values:
        - RANDOM: Uses random values within configured ranges
        - RECURSIVE: Uses RecursiveEmotionProcessor for advanced analysis
        - HYBRID: Combines strategies based on processor availability

        Args:
            words: List of words to process

        Raises:
            EmotionProcessingError: If emotion assignment fails
        """
        for word in words:
            try:
                # Select strategy for emotion assignment
                if (
                    self._strategy == EmotionAssignmentStrategy.RECURSIVE
                    and self._processor
                ):
                    # Use recursive processor for advanced analysis
                    self._assign_recursive_emotion(word)
                elif (
                    self._strategy == EmotionAssignmentStrategy.HYBRID
                    and self._processor
                ):
                    # Use processor with fallback to random
                    self._assign_hybrid_emotion(word)
                else:
                    # Default to random assignment
                    self._assign_random_emotion(word)

                # Update processing counters
                self._state.increment_processed()

                # Log at appropriate level (debug for most, periodically at info)
                if self._state.processed_count % 10 == 0:
                    self.log_info(
                        f"Processed {self._state.processed_count} words so far"
                    )
                else:
                    self.log_debug(f"Tagged {word.term}")

            except Exception as e:
                error_msg = f"Failed to process word {word.term}: {str(e)}"
                self.log_error(error_msg)
                raise EmotionProcessingError(error_msg) from e

    def _assign_random_emotion(self, word: WordEmotion) -> None:
        """
        Assign random emotional values to a word.

        Args:
            word: WordEmotion object to process
        """
        # Get the emotional valence/arousal ranges from config
        valence_range = getattr(config.emotion, "valence_range", (-1.0, 1.0))
        arousal_range = getattr(config.emotion, "arousal_range", (0.0, 1.0))

        # Generate values within the configured ranges
        valence = random.uniform(valence_range[0], valence_range[1])
        arousal = random.uniform(arousal_range[0], arousal_range[1])

        # Store in database
        self.emotion_manager.set_word_emotion(word.word_id, valence, arousal)

        self.log_debug(
            f"Random emotion for {word.term}: valence={valence:.2f}, arousal={arousal:.2f}"
        )

    def _assign_recursive_emotion(self, word: WordEmotion) -> None:
        """
        Assign emotional values using recursive emotion processor.

        Args:
            word: WordEmotion object to process

        Raises:
            EmotionProcessingError: If recursive processing fails
        """
        if not self._processor:
            raise EmotionProcessingError("Recursive processor not available")

        try:
            # Process the term with recursion
            concept = self._processor.process_term(word.term)

            # Extract primary emotion dimensions
            valence = concept.primary_emotion.dimensions.get(
                EmotionDimension.VALENCE, 0.0
            )
            arousal = concept.primary_emotion.dimensions.get(
                EmotionDimension.AROUSAL, 0.0
            )

            # Validate confidence
            if concept.primary_emotion.confidence < self.confidence_threshold:
                self.log_warning(
                    f"Skipping {word.term} due to low confidence: {concept.primary_emotion.confidence:.2f}"
                )
                return

            # Store in database
            self.emotion_manager.set_word_emotion(word.word_id, valence, arousal)

            # Log meta-emotions if debug is enabled
            if self._debug_meta_emotions and concept.meta_emotions:
                self.log_debug(
                    f"Meta-emotions for {word.term}: "
                    f"{', '.join(label for label, _ in concept.meta_emotions)}"
                )

            self.log_debug(
                f"Recursive emotion for {word.term}: valence={valence:.2f}, arousal={arousal:.2f}, "
                f"confidence={concept.primary_emotion.confidence:.2f}"
            )

        except Exception as e:
            self.log_error(f"Recursive processing failed for {word.term}: {e}")
            raise EmotionProcessingError(f"Recursive processing failed: {e}") from e

    def _assign_hybrid_emotion(self, word: WordEmotion) -> None:
        """
        Assign emotional values using hybrid approach.

        Tries recursive processing first, falls back to random if it fails.

        Args:
            word: WordEmotion object to process
        """
        try:
            # First try recursive processing
            self._assign_recursive_emotion(word)
        except Exception as e:
            # Log the fallback
            self.log_warning(f"Falling back to random for {word.term}: {e}")

            # Fall back to random
            self._assign_random_emotion(word)

    def _handle_processing_error(self, error: Exception) -> None:
        """
        Log processing errors with contextual information and update state.

        Args:
            error: The exception that occurred
        """
        error_type = type(error).__name__
        self._state.record_error(error_type)

        # Log the full error with traceback
        self.log_error(f"{error_type}: {str(error)}")
        self.log_debug(traceback.format_exc())

    @eidosian()
    def log_debug(self, message: str) -> None:
        """Log debug message if logging is enabled."""
        if self.enable_logging:
            self.logger.debug(message)

    @eidosian()
    def log_info(self, message: str) -> None:
        """Log info message if logging is enabled."""
        if self.enable_logging:
            self.logger.info(message)

    @eidosian()
    def log_warning(self, message: str) -> None:
        """Log warning message if logging is enabled."""
        if self.enable_logging:
            self.logger.warning(message)

    @eidosian()
    def log_error(self, message: str) -> None:
        """Log error message if logging is enabled."""
        if self.enable_logging:
            self.logger.error(message)

    def _update_state(self, action: str):
        """Context manager to update worker state with actions."""

        class StateUpdater:
            def __init__(self, worker: "EmotionWorker") -> None:
                self.worker = worker
                self.action = action

            def __enter__(self) -> "StateUpdater":
                if self.action == "start":
                    self.worker._state.start()
                elif self.action == "stop":
                    self.worker._state.stop()
                elif self.action == "pause":
                    self.worker._state.pause()
                elif self.action == "resume":
                    self.worker._state.resume()
                elif self.action == "recovery":
                    self.worker._state.record_recovery()
                return self

            def __exit__(
                self,
                exc_type: Optional[Type[BaseException]],
                exc_val: Optional[BaseException],
                exc_tb: Optional[TracebackType],
            ) -> None:
                # Context manager cleanup (no action needed)
                pass

        return StateUpdater(self)
