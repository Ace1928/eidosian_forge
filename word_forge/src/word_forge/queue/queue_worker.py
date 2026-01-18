"""
Worker Processor Module

This module provides a processor implementation for the QueueManager system,
specifically designed to process lexical items. It features:
- Full implementation of the QueueProcessor protocol
- Integration with database and parser components
- Comprehensive error handling with the Result pattern
- Performance metrics and monitoring

The WordProcessor class serves as a bridge between the queue management
system and the lexical processing pipeline.
"""

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from word_forge.database.database_manager import DBManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueProcessor, Result


class ProcessingError(Exception):
    """Base exception for word processing errors with comprehensive context tracking."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        processing_context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.processing_context = processing_context or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        error_msg = self.message
        if self.cause:
            error_msg += f" | Cause: {str(self.cause)}"
        if self.processing_context:
            context_str = ", ".join(
                f"{k}={v}" for k, v in self.processing_context.items()
            )
            error_msg += f" | Context: {{{context_str}}}"
        return error_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "message": self.message,
            "cause": str(self.cause) if self.cause else None,
            "context": self.processing_context,
            "timestamp": self.timestamp,
        }


class ProcessingStatus(Enum):
    """Status of a word processing operation."""

    SUCCESS = auto()
    DUPLICATE = auto()
    DATABASE_ERROR = auto()
    PARSER_ERROR = auto()
    VALIDATION_ERROR = auto()
    GENERAL_ERROR = auto()


@dataclass
class ProcessingResult:
    """
    Result of a word processing operation implementing the Result monad pattern.

    Represents either successful completion with metrics or failure with context.
    This class follows functional programming principles to avoid exceptions
    across component boundaries.

    Attributes:
        status: The processing status (success or specific error)
        term: The term that was processed
        duration_ms: Processing time in milliseconds
        relationships_count: Number of relationships discovered
        new_terms_count: Number of new terms discovered
        error_message: Error message if processing failed
        relationship_types: Dictionary mapping relationship types to counts
    """

    status: ProcessingStatus
    term: str
    duration_ms: float = 0
    relationships_count: int = 0
    new_terms_count: int = 0
    error_message: Optional[str] = None
    relationship_types: Dict[str, int] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if processing was successful."""
        return self.status == ProcessingStatus.SUCCESS

    @property
    def is_duplicate(self) -> bool:
        """Check if term was a duplicate."""
        return self.status == ProcessingStatus.DUPLICATE

    def map(self, f: Callable[[str], str]) -> "ProcessingResult":
        """Apply transformation to term if processing was successful."""
        if not self.is_success:
            return self

        new_result = ProcessingResult(
            status=self.status,
            term=f(self.term),
            duration_ms=self.duration_ms,
            relationships_count=self.relationships_count,
            new_terms_count=self.new_terms_count,
            relationship_types=self.relationship_types,
        )
        return new_result

    @classmethod
    def success(
        cls,
        term: str,
        duration_ms: float,
        relationships_count: int,
        new_terms_count: int,
        relationship_types: Dict[str, int],
    ) -> "ProcessingResult":
        """Create a successful processing result."""
        return cls(
            status=ProcessingStatus.SUCCESS,
            term=term,
            duration_ms=duration_ms,
            relationships_count=relationships_count,
            new_terms_count=new_terms_count,
            relationship_types=relationship_types,
        )

    @classmethod
    def duplicate(cls, term: str) -> "ProcessingResult":
        """Create a result for duplicate terms."""
        return cls(
            status=ProcessingStatus.DUPLICATE,
            term=term,
            error_message=f"Term '{term}' has already been processed",
        )

    @classmethod
    def error(
        cls, term: str, status: ProcessingStatus, message: str
    ) -> "ProcessingResult":
        """Create a result for processing errors."""
        return cls(status=status, term=term, error_message=message)


@dataclass
class ProcessingStats:
    """
    Statistics about word processing operations.

    Tracks performance metrics and processing outcomes
    for monitoring and optimization.

    Attributes:
        processed_count: Total number of terms processed
        success_count: Number of successfully processed terms
        duplicate_count: Number of duplicate terms encountered
        error_count: Number of errors encountered
        total_duration_ms: Total processing time in milliseconds
        start_time: Time when processing started
        last_processed: Last term processed
        relationship_counts: Categorized relationship counts
    """

    processed_count: int = 0
    success_count: int = 0
    duplicate_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0
    start_time: float = field(default_factory=time.time)
    last_processed: Optional[str] = None
    relationship_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_processing_time_ms(self) -> float:
        """Calculate average processing time in milliseconds."""
        if self.processed_count == 0:
            return 0.0
        return self.total_duration_ms / self.processed_count

    @property
    def processing_rate_per_minute(self) -> float:
        """Calculate processing rate in items per minute."""
        elapsed = time.time() - self.start_time
        if elapsed < 0.001:  # Avoid division by zero
            return 0.0
        return (self.processed_count / elapsed) * 60

    def update(self, result: ProcessingResult) -> None:
        """Update statistics with a new processing result."""
        self.processed_count += 1
        self.last_processed = result.term

        if result.is_success:
            self.success_count += 1
            self.total_duration_ms += result.duration_ms

            # Update relationship counts
            for rel_type, count in result.relationship_types.items():
                if rel_type in self.relationship_counts:
                    self.relationship_counts[rel_type] += count
                else:
                    self.relationship_counts[rel_type] = count
        elif result.is_duplicate:
            self.duplicate_count += 1
        else:
            self.error_count += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.processed_count = 0
        self.success_count = 0
        self.duplicate_count = 0
        self.error_count = 0
        self.total_duration_ms = 0
        self.start_time = time.time()
        self.last_processed = None
        self.relationship_counts.clear()


class WordProcessor(QueueProcessor[str]):
    """
    Processes terms from the queue into the lexical database.

    Implements the QueueProcessor protocol to handle lexical
    terms pulled from the queue, updating the database,
    and discovering new relationships.

    Attributes:
        db_manager: Database manager for persistence
        parser_refiner: Parser for lexical processing
        stats: Processing statistics and metrics
    """

    def __init__(
        self,
        db_manager: DBManager,
        parser_refiner: ParserRefiner,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the word processor.

        Args:
            db_manager: Database manager for storage operations
            parser_refiner: Parser for lexical processing
            logger: Optional logger for detailed logging
        """
        self.db_manager = db_manager
        self.parser_refiner = parser_refiner
        self.logger = logger or logging.getLogger(__name__)
        self.stats = ProcessingStats()
        self._lock = threading.RLock()

    def process(self, item: str) -> Result[bool]:
        """
        Process a term from the queue.

        Implements the QueueProcessor protocol method to process
        lexical terms, updating the database with definitions,
        usage examples, and relationships.

        Args:
            item: The term to process

        Returns:
            Result indicating success (True) or failure
        """
        if not item:
            return Result[bool].failure(
                "INVALID_TERM",
                "Term must be a non-empty string",
                {"term_type": str(type(item))},
            )

        start_time = time.time()
        try:
            # Process the term
            processing_result = self._process_term(item)

            # Update statistics
            with self._lock:
                self.stats.update(processing_result)

            # Log the result
            if processing_result.is_success:
                self.logger.info(
                    f"Processed '{item}' in {processing_result.duration_ms:.1f}ms "
                    f"(found {processing_result.relationships_count} relationships, "
                    f"discovered {processing_result.new_terms_count} new terms)"
                )
            elif processing_result.is_duplicate:
                self.logger.debug(f"Skipped duplicate term '{item}'")
            else:
                self.logger.warning(
                    f"Failed to process '{item}': {processing_result.error_message}"
                )

            # Return success if processing succeeded or term was duplicate
            # Fix: Explicitly provide bool type parameter to Result.success
            return Result[bool].success(processing_result.is_success)

        except Exception as e:
            error_message = f"Unexpected error processing term '{item}': {str(e)}"
            self.logger.error(error_message, exc_info=True)

            # Update error statistics
            with self._lock:
                self.stats.error_count += 1

            # Fix: Explicitly provide bool type parameter to Result.failure
            return Result[bool].failure(
                "PROCESSING_ERROR",
                error_message,
                {"term": item, "error_type": str(type(e).__name__)},
            )
        finally:
            # Log performance metrics for long-running operations
            duration_ms = (time.time() - start_time) * 1000
            if duration_ms > 1000:  # Log slow operations (>1s)
                self.logger.info(f"Processing '{item}' took {duration_ms:.1f}ms")

    def _process_term(self, term: str) -> ProcessingResult:
        # Basic validation
        if not term:
            return ProcessingResult.error(
                term, ProcessingStatus.VALIDATION_ERROR, "Empty term provided"
            )

        # Record processing start time
        start_time = time.time()
        relationships_count = 0
        relationship_types: Dict[str, int] = {}

        try:
            # Get queue size before processing
            initial_queue_size = self.parser_refiner.queue_manager.size

            # Process the term
            self.parser_refiner.process_word(term)

            # Calculate new terms found (use property, not method call)
            new_terms_count = (
                self.parser_refiner.queue_manager.size - initial_queue_size
            )

            # Get information about any relationships
            with contextlib.suppress(Exception):
                # Use get_word_entry to safely handle missing terms
                entry = self.db_manager.get_word_entry(term)
                if entry:
                    # Extract relationships and pass as Dict[str, Any]
                    relationships = entry.get("relationships", [])
                    relationships_count = len(relationships)
                    relationship_types = self._categorize_relationships(relationships)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Return successful result
            return ProcessingResult.success(
                term=term,
                duration_ms=duration_ms,
                relationships_count=relationships_count,
                new_terms_count=new_terms_count,
                relationship_types=relationship_types,
            )

        except Exception as e:
            self.logger.warning(f"Failed to process '{term}': {str(e)}")
            return ProcessingResult.error(
                term, ProcessingStatus.GENERAL_ERROR, f"Error processing term: {str(e)}"
            )

    def _categorize_relationships(self, relationships: List[Any]) -> Dict[str, int]:
        """
        Categorize and count relationship types.

        Args:
            relationships: Sequence of relationship dictionaries

        Returns:
            Dictionary mapping relationship types to counts
        """
        type_counts: Dict[str, int] = {}

        for rel in relationships:
            rel_type = rel.get("relationship_type", "unknown")
            if rel_type in type_counts:
                type_counts[rel_type] += 1
            else:
                type_counts[rel_type] = 1

        return type_counts

    def get_statistics(self) -> Dict[str, Union[int, float, str, None, Dict[str, int]]]:
        """
        Get comprehensive processing statistics.

        Returns a dictionary of metrics about processing performance
        and outcomes.

        Returns:
            Dictionary of processing statistics
        """
        with self._lock:
            elapsed_time = time.time() - self.stats.start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            return {
                "processed_count": self.stats.processed_count,
                "success_count": self.stats.success_count,
                "duplicate_count": self.stats.duplicate_count,
                "error_count": self.stats.error_count,
                "avg_processing_time_ms": self.stats.avg_processing_time_ms,
                "processing_rate_per_minute": self.stats.processing_rate_per_minute,
                "runtime": f"{minutes}:{seconds:02d}",
                "runtime_seconds": elapsed_time,
                "last_processed": self.stats.last_processed,
                "relationship_counts": self.stats.relationship_counts,
            }

    def reset_statistics(self) -> None:
        """Reset all processing statistics."""
        with self._lock:
            self.stats.reset()


@dataclass
class WorkerPoolConfig:
    """
    Configuration for the worker pool.

    Contains parameters for controlling the worker pool behavior,
    including size, timeout, and backoff strategies.

    Attributes:
        worker_count: Number of parallel worker threads
        max_queue_size: Maximum queue size before backpressure
        processing_timeout_ms: Maximum time for processing a term
        backoff_factor: Multiplier for exponential backoff
        max_retry_count: Maximum number of retries for failed tasks
    """

    worker_count: int = 4
    max_queue_size: int = 1000
    processing_timeout_ms: int = 30000
    backoff_factor: float = 1.5
    max_retry_count: int = 3


class ParallelProcessorStatus(TypedDict):
    """Status information for parallel word processor."""

    active: bool
    worker_count: int
    available_workers: int
    queue_size: int
    stats: Dict[str, Any]


class ParallelWordProcessor:
    """
    Manages parallel processing of terms using a thread pool.

    Distributes processing work across multiple worker threads
    for improved throughput and resource utilization.

    Attributes:
        processor: The word processor to use for term processing
        config: Configuration for the worker pool
        stats: Aggregate processing statistics
    """

    def __init__(
        self,
        processor: WordProcessor,
        config: Optional[WorkerPoolConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the parallel word processor.

        Args:
            processor: Word processor for handling terms
            config: Optional worker pool configuration
            logger: Optional logger for operational logging
        """
        self.processor = processor
        self.config = config or WorkerPoolConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._active_count = threading.Semaphore(self.config.worker_count)
        self._lock = threading.RLock()
        self._active = False

    def start(self) -> None:
        """
        Start the parallel processing pool.

        Launches worker threads according to configuration
        and begins processing terms from the queue.
        """
        with self._lock:
            if self._active:
                return  # Already running

            self._active = True
            self._stop_event.clear()

            # Launch worker threads
            for i in range(self.config.worker_count):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"WordProcessor-Worker-{i}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()

            self.logger.info(
                f"Started parallel word processor with {self.config.worker_count} workers"
            )

    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """
        Stop all worker threads.

        Args:
            wait: Wait for workers to terminate if True
            timeout: Maximum time to wait for each worker
        """
        if not self._active:
            return

        self._stop_event.set()
        self._active = False

        if wait:
            for worker in self._workers:
                worker.join(timeout)

        self._workers = []
        self.logger.info("Stopped parallel word processor")

    def _worker_loop(self) -> None:
        """
        Main worker thread processing loop.

        Continuously processes terms from the queue until stopped,
        handling errors and backoff according to configuration.
        """
        queue_manager = self.processor.parser_refiner.queue_manager

        while not self._stop_event.is_set():
            # Acquire worker slot
            if not self._active_count.acquire(blocking=False):
                # No slots available, wait briefly
                time.sleep(0.01)
                continue

            try:
                # Dequeue term
                result = queue_manager.dequeue(block=True, timeout=0.5)

                if result.is_failure:
                    # No terms available or queue not active
                    continue

                # Process the term
                term = result.unwrap()
                self.processor.process(term)

            except Exception as e:
                self.logger.error(f"Worker error: {str(e)}", exc_info=True)
            finally:
                # Release worker slot
                self._active_count.release()

    def get_status(self) -> ParallelProcessorStatus:
        """
        Get the current status of the parallel processor.

        Returns:
            Dictionary containing operational status and metrics
        """
        processor_stats = self.processor.get_statistics()

        return {
            "active": self._active,
            "worker_count": self.config.worker_count,
            "available_workers": self._active_count._value,
            # Access size as a property, not a method
            "queue_size": self.processor.parser_refiner.queue_manager.size,
            "stats": processor_stats,
        }


# Export public elements
__all__ = [
    "WordProcessor",
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingStats",
    "ProcessingError",
    "ParallelWordProcessor",
    "WorkerPoolConfig",
]
