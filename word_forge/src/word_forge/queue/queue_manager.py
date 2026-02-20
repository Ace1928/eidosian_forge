from eidosian_core import eidosian

"""
Queue Manager Module

This module provides a thread-safe, generic queue implementation for managing
tasks in the Word Forge system. It features:
- Type-safe generic queue with configurable item types
- Thread-safe operations for concurrent access
- Seen item tracking to prevent duplicate processing
- Priority-based scheduling for important tasks
- Performance metrics and telemetry
- Error handling using Result pattern

The QueueManager class serves as a central component for managing processing
tasks with proper concurrency control and instrumentation.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    Any,  # Keep Any for ErrorContext context
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    TypedDict,
    TypeVar,
    Union,
    cast,  # Keep cast for Result.unwrap
    final,
)

# Type variable for queue items - allows for generic queue
T = TypeVar("T")
R = TypeVar("R")
# Contravariant type variable for QueueProcessor
T_contra = TypeVar("T_contra", contravariant=True)


class QueueError(Exception):
    """
    Base exception for queue operations.

    Provides a consistent foundation for all queue-related exceptions
    with support for capturing the original cause.

    Attributes:
        message: Detailed error description
        cause: Original exception that triggered this error
    """

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        """
        Initialize with detailed error message and optional cause.

        Args:
            message: Error description with context
            cause: Original exception that caused this error (if applicable)
        """
        super().__init__(message)
        self.__cause__ = cause
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        """Provide detailed error message including cause if available."""
        error_msg = self.message
        if self.cause:
            error_msg += f" | Cause: {str(self.cause)}"
        return error_msg


class EmptyQueueError(QueueError):
    """
    Raised when attempting to dequeue from an empty queue.

    This exception is raised when trying to retrieve an item from an empty queue
    in a non-blocking manner, providing clear feedback about the queue state.
    """

    def __init__(self) -> None:
        """Initialize with standard empty queue message."""
        super().__init__("Queue is empty, no items to dequeue")


class QueueFullError(QueueError):
    """
    Raised when attempting to enqueue to a full queue with a size limit.

    This exception provides information about queue capacity limits when
    an enqueue operation cannot be completed due to reaching maximum size.

    Attributes:
        max_size: Maximum size of the queue
    """

    def __init__(self, max_size: int) -> None:
        """
        Initialize with queue full details.

        Args:
            max_size: Maximum capacity of the queue
        """
        super().__init__(f"Queue is full (max size: {max_size})")
        self.max_size = max_size


class TaskPriority(IntEnum):
    """
    Priority levels for queue items.

    Defines the relative importance of tasks in the queue,
    affecting processing order.
    """

    HIGH = auto()
    NORMAL = auto()
    LOW = auto()


@dataclass(order=True)
class PrioritizedItem(Generic[T]):
    """
    Wraps a queue item with priority information.

    This class enables priority-based ordering in the queue
    while preserving the original item data.

    Attributes:
        priority: The item's processing priority
        timestamp: When the item was added (for FIFO within priority)
        item: The actual task data
    """

    priority: TaskPriority = field(compare=True)
    timestamp: float = field(compare=True)
    item: T = field(compare=False)


@dataclass
class QueueMetrics:
    """
    Metrics tracking queue performance and state.

    Collects comprehensive statistics about queue operations
    for monitoring and optimization.

    Attributes:
        enqueued_count: Total items successfully added
        dequeued_count: Total items successfully retrieved
        rejected_count: Items rejected as duplicates
        error_count: Number of errors during operations
        last_enqueued: Last successfully added item
        last_dequeued: Last successfully retrieved item
        avg_wait_time_ms: Average time items spend in queue
        high_priority_count: Number of high-priority items processed
        normal_priority_count: Number of normal-priority items processed
        low_priority_count: Number of low-priority items processed
    """

    enqueued_count: int = 0
    dequeued_count: int = 0
    rejected_count: int = 0
    error_count: int = 0
    last_enqueued: Optional[str] = None
    last_dequeued: Optional[str] = None
    avg_wait_time_ms: float = 0.0
    high_priority_count: int = 0
    normal_priority_count: int = 0
    low_priority_count: int = 0
    wait_times: List[float] = field(default_factory=list)

    @eidosian()
    def update_wait_time(self, wait_time_ms: float) -> None:
        """
        Update the average wait time with a new data point.

        Args:
            wait_time_ms: The wait time in milliseconds to incorporate
        """
        self.wait_times.append(wait_time_ms)
        if len(self.wait_times) > 100:  # Keep a rolling window
            self.wait_times.pop(0)
        self.avg_wait_time_ms = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0.0

    @eidosian()
    def increment_by_priority(self, priority: TaskPriority) -> None:
        """
        Increment the counter for a specific priority level.

        Args:
            priority: The priority level to increment
        """
        if priority == TaskPriority.HIGH:
            self.high_priority_count += 1
        elif priority == TaskPriority.NORMAL:
            self.normal_priority_count += 1
        elif priority == TaskPriority.LOW:
            self.low_priority_count += 1


class QueueState(Enum):
    """
    Represents the current operational state of the queue.

    Used to control the queue's behavior and lifecycle.
    """

    INITIALIZED = auto()  # Initial state after creation
    RUNNING = auto()  # Normal operation, accepting and processing items
    PAUSED = auto()  # Temporarily suspended, not processing new items
    STOPPING = auto()  # In the process of stopping, finishing current work
    STOPPED = auto()  # Completely stopped, no new items processed


@dataclass
class ErrorContext:
    """
    Context information for error handling.

    Provides standardized error information for debugging and monitoring.

    Attributes:
        error_code: Machine-readable error code
        message: Human-readable error message
        timestamp: When the error occurred
        context: Additional contextual information
    """

    error_code: str
    message: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Result(Generic[T]):
    """
    Represents the outcome of an operation that may fail.

    Implements the Result pattern for safe error handling without exceptions.

    Attributes:
        value: The successful result value (if operation succeeded)
        error: Error context (if operation failed)
    """

    value: Optional[T] = None
    error: Optional[ErrorContext] = None

    @property
    def is_success(self) -> bool:
        """Check if the operation was successful."""
        return self.error is None

    @property
    def is_failure(self) -> bool:
        """Check if the operation failed."""
        return self.error is not None

    @eidosian()
    def unwrap(self) -> T:
        """
        Safely extract the success value, raising an exception if failed.

        Returns:
            The successful operation result

        Raises:
            ValueError: If trying to unwrap a failed result
        """
        if self.is_failure:
            error_msg = f"Cannot unwrap failed result: {self.error.message if self.error else 'Unknown error'}"
            raise ValueError(error_msg)
        return cast(T, self.value)

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create a successful result with the given value."""
        return cls(value=value)

    @classmethod
    def failure(cls, error_code: str, message: str, context: Optional[Dict[str, Any]] = None) -> "Result[T]":
        """Create a failed result with error information."""
        return cls(error=ErrorContext(error_code=error_code, message=message, context=context or {}))


class QueueProcessor(Protocol, Generic[T_contra]):
    """
    Protocol defining the interface for queue item processors.

    Ensures type safety when working with queue processors.
    """

    @eidosian()
    def process(self, item: T_contra) -> Result[bool]:
        """
        Process a queue item.

        Args:
            item: The item to process

        Returns:
            Result indicating success or failure with error context
        """
        ...


@final
class QueueManager(Generic[T]):
    """
    Thread-safe queue manager for processing tasks.

    Manages a queue of items with priority support, duplicate detection,
    and comprehensive metrics. Designed for concurrent access with proper
    synchronization.

    Attributes:
        size_limit: Maximum number of items allowed in queue
        state: Current operational state
        metrics: Performance and operational metrics
    """

    def __init__(self, size_limit: int = 0) -> None:
        """
        Initialize the queue manager.

        Args:
            size_limit: Maximum number of items allowed in queue (0 for unlimited)

        Examples:
            >>> # Unlimited queue
            >>> queue_manager = QueueManager()
            >>> # Limited queue with max 1000 items
            >>> limited_queue = QueueManager(size_limit=1000)
        """
        self._queue: queue.PriorityQueue[PrioritizedItem[T]] = queue.PriorityQueue(maxsize=size_limit)
        self._seen_items: Set[str] = set()
        self._lock = threading.RLock()
        self._state = QueueState.INITIALIZED
        self._size_limit = size_limit
        self._entry_times: Dict[str, float] = {}

        # Public attributes
        self.metrics = QueueMetrics()

    def __repr__(self) -> str:
        """Return a string representation of the queue manager."""
        return f"<QueueManager state={self.state.name} size={self.size}>"

    def __len__(self) -> int:
        """Return the current size of the queue."""
        return self.size

    def __bool__(self) -> bool:
        """Check if the queue is empty."""
        return not self.is_empty

    def __iter__(self) -> Iterator[PrioritizedItem[T]]:
        """Return an iterator over the queue items."""
        with self._lock:
            return iter(self._queue.queue)

    def __iter_seen__(self) -> Iterator[str]:
        """Return an iterator over the seen items keys."""
        with self._lock:
            return iter(self._seen_items)

    @eidosian()
    def seen_items(self) -> List[str]:
        """Return a list of seen item keys."""
        with self._lock:
            return list(self._seen_items)

    @eidosian()
    def iter_seen(self) -> Iterator[str]:
        """Return an iterator over the seen item keys."""
        with self._lock:
            return iter(self._seen_items)

    @property
    def size(self) -> int:
        """
        Get the current number of items in the queue.

        Returns:
            The number of items currently in the queue
        """
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """
        Check if the queue is currently empty.

        Returns:
            True if queue contains no items, False otherwise
        """
        return self._queue.empty()

    @property
    def state(self) -> QueueState:
        """
        Get the current queue state.

        Returns:
            Current operational state of the queue
        """
        with self._lock:
            return self._state

    @state.setter
    def state(self, new_state: QueueState) -> None:
        """
        Set the queue to a new state.

        Args:
            new_state: The state to transition to
        """
        with self._lock:
            self._state = new_state

    def _item_to_key(self, item: T) -> str:
        """
        Convert an item to a unique string key for deduplication.

        Args:
            item: The item to convert

        Returns:
            A string key representing the item
        """
        # Handle strings directly
        if isinstance(item, str):
            return item.strip().lower()

        # Handle objects with a proper string representation
        return str(item).strip().lower()

    @eidosian()
    def enqueue(self, item: T, priority: TaskPriority = TaskPriority.NORMAL) -> Result[bool]:
        """
        Add an item to the queue if not already present.

        Thread-safe method to add an item to the priority queue
        with duplicate detection.

        Args:
            item: The item to add to the queue
            priority: Priority level affecting processing order

        Returns:
            Result indicating success (True if added, False if duplicate) or failure

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> result = queue_manager.enqueue("process_me", TaskPriority.HIGH)
            >>> if result.is_success:
            ...     print("Item added" if result.unwrap() else "Item was duplicate")
        """
        if self.state in (QueueState.STOPPING, QueueState.STOPPED):
            return Result[bool].failure(
                "QUEUE_STOPPED",
                "Queue is not accepting new items",
                {"state": self.state.name},
            )

        try:
            item_key = self._item_to_key(item)

            with self._lock:
                # Check if it's already been seen
                if item_key in self._seen_items:
                    self.metrics.rejected_count += 1
                    return Result[bool].success(False)

                # Add to the queue with priority
                prioritized = PrioritizedItem(priority=priority, timestamp=time.time(), item=item)

                # Try to add to the queue
                try:
                    self._queue.put_nowait(prioritized)
                except queue.Full:
                    return Result[bool].failure(
                        "QUEUE_FULL",
                        f"Queue is full (max size: {self._size_limit})",
                        {"size_limit": str(self._size_limit)},
                    )

                # Track the item
                self._seen_items.add(item_key)
                self._entry_times[item_key] = time.time()

                # Update metrics
                self.metrics.enqueued_count += 1
                self.metrics.last_enqueued = str(item)

                return Result[bool].success(True)

        except Exception as e:
            self.metrics.error_count += 1
            return Result[bool].failure(
                "ENQUEUE_ERROR",
                f"Error enqueueing item: {str(e)}",
                {"error_type": type(e).__name__},
            )

    @eidosian()
    def dequeue(self, block: bool = False, timeout: Optional[float] = None) -> Result[T]:
        """
        Get the next item from the queue based on priority.

        Thread-safe method to get the highest priority item from the queue.

        Args:
            block: If True, block until an item is available
            timeout: Maximum time to wait if blocking (None for no timeout)

        Returns:
            Result containing the next item or error information

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> queue_manager.enqueue("item1")
            >>> result = queue_manager.dequeue()
            >>> if result.is_success:
            ...     item = result.unwrap()
            ...     print(f"Processing: {item}")
        """
        if self.state in (QueueState.PAUSED, QueueState.STOPPING, QueueState.STOPPED):
            return Result[T].failure(
                "QUEUE_NOT_ACTIVE",
                f"Queue is not actively processing items (state: {self.state.name})",
                {"state": self.state.name},
            )

        try:
            try:
                # Get the next item
                prioritized: PrioritizedItem[T] = self._queue.get(block=block, timeout=timeout)
                item = prioritized.item

                # Update metrics
                with self._lock:
                    item_key = self._item_to_key(item)
                    entry_time = self._entry_times.pop(item_key, None)

                    self.metrics.dequeued_count += 1
                    self.metrics.last_dequeued = str(item)
                    self.metrics.increment_by_priority(prioritized.priority)

                    # Calculate wait time if we have entry time
                    if entry_time is not None:
                        wait_time_ms = (time.time() - entry_time) * 1000
                        self.metrics.update_wait_time(wait_time_ms)

                return Result[T].success(item)

            except queue.Empty:
                return Result[T].failure("QUEUE_EMPTY", "Queue is empty, no items to dequeue", {})

        except Exception as e:
            self.metrics.error_count += 1
            return Result[T].failure(
                "DEQUEUE_ERROR",
                f"Error dequeuing item: {str(e)}",
                {"error_type": type(e).__name__},
            )

    @eidosian()
    def clear(self) -> Result[int]:
        """
        Clear all items from the queue.

        Thread-safe method to empty the queue and reset tracking.

        Returns:
            Result containing the number of items cleared

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> # Add 10 items
            >>> for i in range(10):
            ...     queue_manager.enqueue(f"item{i}")
            >>> result = queue_manager.clear()
            >>> if result.is_success:
            ...     print(f"Cleared {result.unwrap()} items")
        """
        try:
            with self._lock:
                # Count items before clearing
                count = self.size

                # Clear the underlying queue
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

                # Reset tracking sets and metrics
                self._seen_items.clear()
                self._entry_times.clear()

                return Result[int].success(count)

        except Exception as e:
            self.metrics.error_count += 1
            return Result[int].failure(
                "CLEAR_ERROR",
                f"Error clearing queue: {str(e)}",
                {"error_type": type(e).__name__},
            )

    @eidosian()
    def mark_seen(self, item: T) -> None:
        """
        Mark an item as seen without adding it to the queue.

        Thread-safe method to prevent an item from being added
        to the queue in the future.

        Args:
            item: The item to mark as seen

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> queue_manager.mark_seen("already_processed")
            >>> result = queue_manager.enqueue("already_processed")
            >>> # result.unwrap() will be False since item was marked seen
        """
        with self._lock:
            item_key = self._item_to_key(item)
            self._seen_items.add(item_key)

    @eidosian()
    def is_seen(self, item: T) -> bool:
        """
        Check if an item has been seen before.

        Thread-safe method to check if an item has already been
        processed or is currently in the queue.

        Args:
            item: The item to check

        Returns:
            True if the item has been seen, False otherwise

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> queue_manager.enqueue("test_item")
            >>> queue_manager.is_seen("test_item")
            True
            >>> queue_manager.is_seen("new_item")
            False
        """
        with self._lock:
            item_key = self._item_to_key(item)
            return item_key in self._seen_items

    @eidosian()
    def reset_seen(self) -> None:
        """
        Clear the set of seen items.

        Thread-safe method to reset tracking of previously seen items.

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> queue_manager.enqueue("test_item")
            >>> queue_manager.is_seen("test_item")
            True
            >>> queue_manager.reset_seen()
            >>> queue_manager.is_seen("test_item")
            False
        """
        with self._lock:
            self._seen_items.clear()

    @eidosian()
    def process_with(self, processor: QueueProcessor[T], count: int = 1) -> Result[int]:
        """
        Process a specific number of items using the provided processor.

        Thread-safe method to process multiple items in sequence.

        Args:
            processor: Object implementing the processor interface
            count: Number of items to process (default 1)

        Returns:
            Result containing the number of items successfully processed
        """
        if count < 1:
            return Result[int].failure(
                "INVALID_COUNT",
                "Count must be at least 1",
                {"requested_count": str(count)},
            )

        processed = 0
        failures = 0

        for _ in range(count):
            # Get the next item
            dequeue_result = self.dequeue(block=False)

            if dequeue_result.is_failure:
                # Queue is empty or not active
                break

            # Process the item
            try:
                item = dequeue_result.unwrap()
                process_result = processor.process(item)

                if process_result.is_success and process_result.unwrap():
                    processed += 1
                else:
                    failures += 1
            except Exception:
                failures += 1

        return Result[int].success(processed)

    @eidosian()
    def start(self) -> None:
        """
        Start or resume queue processing.

        Sets the queue state to RUNNING, allowing processing of items.

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> queue_manager.state = QueueState.PAUSED
            >>> # Queue won't process items while PAUSED
            >>> queue_manager.start()
            >>> # Now queue will process items
        """
        self.state = QueueState.RUNNING

    @eidosian()
    def pause(self) -> None:
        """
        Pause queue processing.

        Sets the queue state to PAUSED, preventing further processing
        until resumed.

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> queue_manager.pause()
            >>> # Queue won't process items until start() is called
        """
        self.state = QueueState.PAUSED

    @eidosian()
    def stop(self) -> None:
        """
        Stop queue processing.

        Sets the queue state to STOPPED, preventing any further processing.

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> # After some processing
            >>> queue_manager.stop()
            >>> # Queue won't process any more items
        """
        self.state = QueueState.STOPPED

    @eidosian()
    def get_stats(self) -> Dict[str, Union[int, float, str, None]]:
        """
        Get comprehensive queue statistics.

        Returns a dictionary of queue metrics and state information.

        Returns:
            Dictionary of queue statistics

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> # After some processing
            >>> stats = queue_manager.get_stats()
            >>> print(f"Queue size: {stats['current_size']}")
        """
        with self._lock:
            return {
                "state": self.state.name,
                "current_size": self.size,
                "enqueued_count": self.metrics.enqueued_count,
                "dequeued_count": self.metrics.dequeued_count,
                "rejected_count": self.metrics.rejected_count,
                "error_count": self.metrics.error_count,
                "last_enqueued": self.metrics.last_enqueued,
                "last_dequeued": self.metrics.last_dequeued,
                "avg_wait_time_ms": self.metrics.avg_wait_time_ms,
                "high_priority_count": self.metrics.high_priority_count,
                "normal_priority_count": self.metrics.normal_priority_count,
                "low_priority_count": self.metrics.low_priority_count,
            }

    @eidosian()
    def get_sample(self, count: int = 10) -> Result[List[T]]:
        """
        Get a sample of items from the queue without removing them.

        Thread-safe method to retrieve a subset of queue items for inspection.
        Does not affect queue state or processing order.

        Args:
            count: Maximum number of items to retrieve (default 10)

        Returns:
            Result containing a list of queue items or error information

        Examples:
            >>> queue_manager = QueueManager[str]()
            >>> for i in range(5):
            ...     queue_manager.enqueue(f"item{i}")
            >>> result = queue_manager.get_sample(3)
            >>> if result.is_success:
            ...     items = result.unwrap()
            ...     for item in items:
            ...         print(f"Queue contains: {item}")
        """
        if count < 1:
            return Result[List[T]].failure(
                "INVALID_COUNT",
                "Sample count must be at least 1",
                {"requested_count": str(count)},
            )

        try:
            with self._lock:
                # If queue is empty, return empty list
                if self._queue.empty():
                    return Result[List[T]].success([])

                # Make a copy of the queue items without removing them
                # We need to be careful to maintain the original queue state
                items: List[T] = []
                temp_storage: List[PrioritizedItem[T]] = []

                # Get items from the queue (temporarily)
                sample_count = min(count, self.size)
                for _ in range(sample_count):
                    if self._queue.empty():
                        break
                    try:
                        prioritized: PrioritizedItem[T] = self._queue.get_nowait()
                        items.append(prioritized.item)
                        temp_storage.append(prioritized)
                    except queue.Empty:
                        break

                # Put all items back in the queue in their original order
                for prioritized in temp_storage:
                    self._queue.put(prioritized)

                return Result[List[T]].success(items)

        except Exception as e:
            self.metrics.error_count += 1
            return Result[List[T]].failure(
                "SAMPLE_ERROR",
                f"Error retrieving queue sample: {str(e)}",
                {"error_type": type(e).__name__},
            )


class WorkerMetrics(TypedDict):
    """
    Type definition for work distributor metrics.

    Defines the structure of the metrics dictionary used in WorkDistributor.
    """

    items_processed: int
    errors: int
    start_time: float
    worker_states: Dict[int, str]  # State description is string
    last_processed_item: Optional[str]


class WorkDistributor(Generic[T]):
    """
    Manages parallel processing of queue items.

    Distributes queue processing work across multiple worker threads
    with load balancing and performance monitoring.

    Attributes:
        queue_manager: The queue manager containing items to process
        max_workers: Maximum number of parallel workers
        metrics: Processing metrics and statistics
    """

    def __init__(
        self,
        queue_manager: QueueManager[T],
        processor: QueueProcessor[T],
        max_workers: int = 4,
    ) -> None:
        """
        Initialize the work distributor with a queue and processor.

        Args:
            queue_manager: The queue to process items from
            processor: The processor to handle queue items
            max_workers: Maximum number of concurrent worker threads
        """
        self.queue_manager = queue_manager
        self.processor = processor
        self.max_workers = max_workers
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.metrics: WorkerMetrics = {
            "items_processed": 0,
            "errors": 0,
            "start_time": 0.0,
            "worker_states": {},
            "last_processed_item": None,
        }

    @eidosian()
    def start_processing(self) -> None:
        """
        Start parallel processing of queue items.

        Launches worker threads to process items from the queue manager.
        """
        with self._lock:
            if self._workers:
                return  # Already started

            self._stop_event.clear()
            self.metrics["start_time"] = time.time()

            # Create and start workers
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"QueueWorker-{i}",
                    args=(i,),
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()
                self.metrics["worker_states"][i] = "started"

    @eidosian()
    def stop_processing(self, timeout: float = 5.0) -> None:
        """
        Stop all processing workers.

        Args:
            timeout: Maximum time to wait for workers to stop
        """
        self._stop_event.set()

        with self._lock:
            # Wait for workers to finish
            for i, worker in enumerate(self._workers):
                if worker.is_alive():
                    self.metrics["worker_states"][i] = "stopping"
                    worker.join(timeout=timeout)
                    self.metrics["worker_states"][i] = "stopped"

            self._workers = []

    def _worker_loop(self, worker_id: int) -> None:
        """
        Main worker thread loop.

        Continuously processes items from the queue until stopped.

        Args:
            worker_id: Unique identifier for this worker
        """
        while not self._stop_event.is_set():
            # Check if queue is active
            if self.queue_manager.state != QueueState.RUNNING:
                time.sleep(0.1)
                continue

            # Get the next item
            result = self.queue_manager.dequeue(block=True, timeout=0.5)

            if result.is_failure:
                continue

            # Process the item
            try:
                item = result.unwrap()
                self.metrics["worker_states"][worker_id] = f"processing {str(item)[:20]}"

                process_result = self.processor.process(item)

                with self._lock:
                    if process_result.is_success and process_result.unwrap():
                        self.metrics["items_processed"] += 1
                    else:
                        self.metrics["errors"] += 1

            except Exception:
                with self._lock:
                    self.metrics["errors"] += 1
            finally:
                self.metrics["worker_states"][worker_id] = "idle"


# Export public elements
__all__ = [
    "QueueManager",
    "TaskPriority",
    "QueueState",
    "QueueProcessor",
    "Result",
    "QueueError",
    "EmptyQueueError",
    "QueueFullError",
    "WorkDistributor",
]
