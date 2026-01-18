#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance, Offline Audio/Video Transcription Script (Eidosian Refinement v3.14.15)

Provides robust, efficient, offline transcription using OpenAI's Whisper model.

This script offers a command-line and interactive interface for transcribing
various audio and video files locally. It leverages FFmpeg for media handling
and Whisper for transcription, optimized for CPU usage. The implementation
emphasizes type safety, structured error handling via the Result monad,
parallel processing, and comprehensive observability.

Core Principles:
    - **Type Integrity:** Strict typing prevents runtime errors.
    - **Parallelism:** Adaptive concurrency maximizes throughput.
    - **Resilience:** Systematic error handling ensures graceful failure.
    - **Modularity:** Components are reusable and testable.
    - **Retrocompatibility:** Maintains original CLI behavior.
    - **Observability:** Built-in metrics for performance analysis.

Key Features:
    - Supports common audio/video formats (via FFmpeg).
    - Automatic audio extraction from video.
    - Selectable Whisper model sizes (tiny to large).
    - Parallel processing for directories via `ThreadPoolExecutor`.
    - Recursive directory scanning.
    - Interactive mode for user-friendly parameter input.
    - `Result` monad for explicit error propagation.
    - Performance metrics collection (`measure_execution`).
    - Google/Napoleon style docstrings.

Dependencies:
    - Python 3.8+
    - openai-whisper
    - torch (CPU usage enforced)
    - ffmpeg-python
    - typing_extensions (for older Python 3.x versions if needed)

Usage Examples:
    Command Line:
        python av_to_text.py -i video.mp4 -o transcript.txt -m small
        python av_to_text.py -i media_dir/ -o output_dir/ -m base -r -w 4 --stats

    Interactive:
        python av_to_text.py
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from multiprocessing import cpu_count
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

# Conditional imports for graceful failure if dependencies are missing
try:
    import whisper  # type: ignore[import-untyped]
    from whisper import Whisper  # type: ignore[import-untyped]
except ImportError:
    whisper = None
    Whisper = None  # type: ignore[misc, assignment]

try:
    import ffmpeg  # type: ignore[import-untyped]
except ImportError:
    ffmpeg = None

# =============================================================================
# ⚙️ Core Types and Data Structures
# =============================================================================

# --- Type Variables ---
T = TypeVar("T")  # Generic type for Result value
R = TypeVar("R")  # Generic type for function return values or mapped Result
P = TypeVar("P", bound="ProcessFileParams")  # Parameter type for file processing tasks

# --- Domain Specific Types ---
ModelSize = Literal["tiny", "base", "small", "medium", "large"]
VALID_MODEL_SIZES: Set[ModelSize] = {"tiny", "base", "small", "medium", "large"}
_DEFAULT_MODEL_SIZE: ModelSize = "tiny"

AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm"}


class WhisperSegment(TypedDict):
    """
    Represents a single segment from Whisper's transcription output.

    Attributes:
        id: Segment identifier.
        seek: Seek offset.
        start: Start time in seconds.
        end: End time in seconds.
        text: Transcribed text of the segment.
        tokens: List of token IDs.
        temperature: Temperature used for sampling.
        avg_logprob: Average log probability.
        compression_ratio: Compression ratio.
        no_speech_prob: Probability of no speech.
    """

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class WhisperResult(TypedDict):
    """
    Represents the full transcription result structure from Whisper.

    Attributes:
        text: The complete transcribed text.
        segments: A list of detailed transcription segments.
        language: Detected language code.
    """

    text: str
    segments: List[WhisperSegment]
    language: str


# Alias for clarity in function signatures returning transcription data
TranscriptionResult = WhisperResult


# =============================================================================
# 🎯 Error Handling Matrix (Result Monad Pattern)
# =============================================================================


class ErrorSeverity(Enum):
    """Defines the severity level of an error."""

    FATAL = auto()  # System cannot reasonably continue.
    ERROR = auto()  # Operation failed, potentially recoverable.
    WARNING = auto()  # Operation completed but with issues or deviations.
    INFO = auto()  # Informational message, not technically an error.


class ErrorCategory(Enum):
    """Categorizes errors based on their origin or nature."""

    VALIDATION = auto()  # Input data or configuration is invalid.
    RESOURCE = (
        auto()
    )  # Issue accessing or using a required resource (file, network, model).
    EXTERNAL = auto()  # Failure in an external dependency (FFmpeg, Whisper).
    UNEXPECTED = auto()  # An unforeseen error occurred (bug).
    DEPENDENCY = auto()  # A required library or tool is missing.


@dataclass(frozen=True)
class Error:
    """
    Immutable structured error object providing context for failures.

    Attributes:
        message: Human-readable description of the error.
        code: A unique machine-readable code for the error type (e.g., "FILE_NOT_FOUND").
        category: The category classifying the error's nature (e.g., ErrorCategory.VALIDATION).
        severity: The severity level of the error (e.g., ErrorSeverity.ERROR).
        context: Additional key-value pairs providing context (e.g., {"path": "/path/to/file"}).
        trace: Optional traceback string if available, captured for severe errors.
    """

    message: str
    code: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[str] = None

    @classmethod
    def create(
        cls: Type["Error"],
        message: str,
        code: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[BaseException] = None,
    ) -> "Error":
        """
        Factory method to create a standardized Error object.

        Captures traceback automatically for FATAL/ERROR severity or if an
        exception is provided.

        Args:
            message: Human-readable error description.
            code: Machine-readable error code.
            category: The category of the error.
            severity: The severity of the error.
            context: Optional dictionary with contextual information.
            exception: Optional originating exception, used for traceback.

        Returns:
            An initialized Error object.
        """
        trace_str: Optional[str] = None
        if exception:
            trace_str = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
        elif severity in (ErrorSeverity.FATAL, ErrorSeverity.ERROR):
            # Capture traceback for severe errors even without explicit exception
            try:
                # Limit traceback depth? For now, capture full trace.
                trace_str = traceback.format_exc()
                # Clean up the trace to remove this factory method call if desired
                # (Can make traces slightly less noisy but might hide context)
            except Exception:
                trace_str = "Traceback could not be generated."

        return cls(
            message=message,
            code=code,
            category=category,
            severity=severity,
            context=context or {},
            trace=trace_str,
        )


@dataclass(frozen=True)
class Result(Generic[T]):
    """
    Represents the outcome of an operation: either success with a value or failure with an Error.

    This generic container enforces explicit handling of potential failures,
    avoiding reliance on exceptions for expected error conditions.

    Type Parameters:
        T: The type of the value contained in a successful result.

    Attributes:
        value: The result value if the operation was successful (Optional[T]).
        error: An Error object if the operation failed (Optional[Error]).
    """

    value: Optional[T] = None
    error: Optional[Error] = None

    @property
    def is_success(self) -> bool:
        """Returns True if the operation succeeded (has value, no error)."""
        # Check value is not None as well for stricter success definition
        return self.error is None and self.value is not None

    @property
    def is_failure(self) -> bool:
        """Returns True if the operation failed (has error)."""
        return self.error is not None

    def unwrap(self) -> T:
        """
        Returns the success value. Raises ValueError if the result is a failure.

        Use this when failure is considered a programming error or unrecoverable
        at the point of calling.

        Raises:
            ValueError: If the result represents a failure.

        Returns:
            The contained value `T` if successful.
        """
        if self.is_failure or self.value is None:
            error_info = f": {self.error}" if self.error else ""
            raise ValueError(f"Cannot unwrap a failed Result{error_info}")
        return self.value

    def expect(self, message: str) -> T:
        """
        Returns the success value. Raises ValueError with a custom message if failure.

        Similar to `unwrap`, but allows specifying a context-specific error message.

        Args:
            message: The error message to use if unwrapping fails.

        Raises:
            ValueError: If the result represents a failure, including the custom message.

        Returns:
            The contained value `T` if successful.
        """
        if self.is_failure or self.value is None:
            error_info = f": {self.error}" if self.error else ""
            raise ValueError(f"{message}{error_info}")
        return self.value

    def map(self, func: Callable[[T], R]) -> "Result[R]":
        """
        Transforms a `Result[T]` to `Result[R]` by applying `func` to the success value.

        If the original result is a failure, the error is propagated without
        calling `func`. If `func` raises an exception during execution, it's
        caught and returned as a new failure `Result`.

        Args:
            func: A function to apply to the success value `T`, returning `R`.

        Returns:
            A `Result[R]` containing the transformed value or an appropriate error.
        """
        if self.is_failure or self.value is None:
            # Propagate the existing error, ensuring type compatibility.
            # We cast because the error state doesn't hold a value of type R.
            return cast(Result[R], self)
        try:
            new_value: R = func(self.value)
            return Result.success(new_value)
        except Exception as e:
            # Capture exceptions raised by the mapping function
            return Result.failure(
                Error.create(
                    message=f"Mapping function '{func.__name__}' failed: {e}",
                    code="MAP_FUNCTION_ERROR",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                    context={"original_value_repr": repr(self.value)[:100]},
                    exception=e,
                )
            )

    def flat_map(self, func: Callable[[T], "Result[R]"]) -> "Result[R]":
        """
        Transforms `Result[T]` by applying `func` which returns `Result[R]`.

        If the original result is success, `func` is called with the value `T`.
        The `Result[R]` returned by `func` becomes the final result.
        If the original result is failure, the error is propagated.
        Exceptions during `func` execution are caught and returned as failures.
        Also known as 'bind' or 'and_then'.

        Args:
            func: A function accepting `T` and returning `Result[R]`.

        Returns:
            The `Result[R]` produced by `func` or an appropriate error `Result`.
        """
        if self.is_failure or self.value is None:
            return cast(Result[R], self)
        try:
            # The function itself returns a Result
            return func(self.value)
        except Exception as e:
            # Capture exceptions raised by the flat-mapping function
            return Result.failure(
                Error.create(
                    message=f"Flat-mapping function '{func.__name__}' failed: {e}",
                    code="FLATMAP_FUNCTION_ERROR",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                    context={"original_value_repr": repr(self.value)[:100]},
                    exception=e,
                )
            )

    @classmethod
    def success(cls: Type["Result"], value: T) -> "Result[T]":
        """Creates a Result representing a successful operation with the given value."""
        if value is None:
            # While technically allowed by Optional[T], success(None) can be ambiguous.
            # Consider using Result[None] explicitly if None is a valid success value.
            logging.debug("Creating Result.success with None value.")
        return cls(value=value, error=None)

    @classmethod
    def failure(cls: Type["Result"], error: Error) -> "Result[T]":
        """Creates a Result representing a failed operation with the given Error."""
        # The value is None in a failure case. The generic type T remains
        # relevant for type consistency in chains, even though value is None.
        return cls(value=None, error=error)

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the Result."""
        if self.is_success:
            value_repr = repr(self.value)
            if len(value_repr) > 100:
                value_repr = value_repr[:97] + "..."
            return f"Result.success(value={value_repr})"
        else:
            # Error should never be None if is_success is False, but check defensively
            error_repr = repr(self.error) if self.error else "<?>"
            return f"Result.failure(error={error_repr})"


# =============================================================================
# 📊 Observability and Metrics
# =============================================================================


@dataclass
class ExecutionMetrics:
    """
    Holds performance and context metrics for a specific operation execution.

    Attributes:
        operation: Name identifying the measured operation (e.g., "transcribe_audio").
        duration_ns: Execution time in nanoseconds.
        start_time_ns: Timestamp when the operation started (monotonic, nanoseconds).
        context: Dictionary holding contextual information (e.g., file paths, model size).
    """

    operation: str
    duration_ns: int = 0
    start_time_ns: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Returns the duration in milliseconds."""
        return self.duration_ns / 1_000_000


class MetricsRegistry:
    """
    A thread-safe registry for collecting and analyzing ExecutionMetrics.

    Provides methods to record metrics and calculate summary statistics.
    """

    def __init__(self) -> None:
        """Initializes the registry with a lock for thread safety."""
        self._metrics: Dict[str, List[ExecutionMetrics]] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested measurements

    def record(self, metrics: ExecutionMetrics) -> None:
        """
        Records the metrics for a completed operation in a thread-safe manner.

        Args:
            metrics: The ExecutionMetrics object to record.
        """
        with self._lock:
            # Use setdefault for cleaner initialization of the list per operation
            self._metrics.setdefault(metrics.operation, []).append(metrics)

    def get_statistics(self, operation: str) -> Optional[Dict[str, float | int]]:
        """
        Calculates summary statistics for a given operation.

        Args:
            operation: The name of the operation to analyze.

        Returns:
            A dictionary with statistics (count, avg_duration_ms, min_duration_ms,
            max_duration_ms, total_duration_ms) or None if no metrics exist.
        """
        with self._lock:
            # Create a copy to analyze without holding the lock for long
            op_metrics = list(self._metrics.get(operation, []))

        if not op_metrics:
            return None

        count = len(op_metrics)
        durations_ns = [m.duration_ns for m in op_metrics]
        total_duration_ns = sum(durations_ns)

        stats: Dict[str, float | int] = {
            "count": count,
            "avg_duration_ms": (total_duration_ns / count) / 1_000_000,
            "min_duration_ms": min(durations_ns) / 1_000_000,
            "max_duration_ms": max(durations_ns) / 1_000_000,
            "total_duration_ms": total_duration_ns / 1_000_000,
        }
        return stats

    def get_all_statistics(self) -> Dict[str, Optional[Dict[str, float | int]]]:
        """
        Calculates summary statistics for all recorded operations.

        Returns:
            A dictionary where keys are operation names and values are their
            statistics dictionaries (or None if no data).
        """
        with self._lock:
            # Get a snapshot of keys to iterate over safely
            all_ops = list(self._metrics.keys())
        return {op: self.get_statistics(op) for op in all_ops}

    def clear(self) -> None:
        """Clears all recorded metrics from the registry."""
        with self._lock:
            self._metrics.clear()


# --- Global Metrics Registry Instance ---
metrics_registry = MetricsRegistry()


@contextmanager
def measure_execution(
    operation: str, context: Optional[Dict[str, Any]] = None
) -> Iterator[ExecutionMetrics]:
    """
    Context manager to measure the execution time of a block of code.

    Automatically records the metrics to the global `metrics_registry` upon
    exiting the context block (whether normally or via exception).

    Args:
        operation: A descriptive name for the operation being measured (e.g., "extract_audio").
        context: Optional dictionary providing context for the measurement (e.g., file paths).

    Yields:
        An ExecutionMetrics object that is populated with timing information
        when the context block exits.

    Example:
        >>> ctx = {"file": "audio.mp3"}
        >>> with measure_execution("audio_processing", ctx) as metrics:
        ...     time.sleep(0.1)
        ...     print(f"Processing took {metrics.duration_ms:.2f} ms") # Duration available after block
        # Metrics are automatically recorded in the registry.
    """
    metrics_obj = ExecutionMetrics(
        operation=operation,
        start_time_ns=time.perf_counter_ns(),
        context=context or {},
    )
    try:
        yield metrics_obj
    finally:
        # Calculate duration and record regardless of exceptions
        metrics_obj.duration_ns = time.perf_counter_ns() - metrics_obj.start_time_ns
        metrics_registry.record(metrics_obj)


# =============================================================================
# ⚡ Parallel Processing Architecture
# =============================================================================


# --- Task Definition ---
class TaskPriority(Enum):
    """Defines priority levels for tasks in the queue."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


# Type variable for Task parameters
TaskParams = TypeVar("TaskParams")
# Type variable for Task return value (which should be a Result)
TaskResultType = TypeVar("TaskResultType", bound=Result[Any])


@dataclass(order=True)
class Task(Generic[TaskParams, TaskResultType]):
    """
    Represents a unit of work with priority for the WorkDistributor.

    The function `func` is expected to return a `Result`.

    Attributes:
        priority: The priority level (lower value means higher priority).
        func: The callable function to execute. Expected signature: `(TaskParams) -> TaskResultType`.
        params: The parameters object or value to pass to the function.
        context: Optional dictionary for tracing, metrics, or other metadata.
    """

    priority: TaskPriority = field(compare=True)
    func: Callable[[TaskParams], TaskResultType] = field(compare=False)
    params: TaskParams = field(compare=False)
    context: Dict[str, Any] = field(default_factory=dict, compare=False)

    def execute(self) -> TaskResultType:
        """Executes the task's function with its parameters."""
        # The function itself is responsible for returning a Result
        return self.func(self.params)


# --- Work Distributor ---
class WorkDistributor:
    """
    Manages a thread pool for executing Tasks concurrently with error handling.

    Integrates with the metrics system and expects task functions to return
    `Result` objects. Provides a `map` function for easy parallel processing
    of iterables.
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        """
        Initializes the WorkDistributor.

        Args:
            max_workers: Max worker threads. Defaults to `cpu_count()`. Must be >= 1.
        """
        resolved_workers = max(1, max_workers or cpu_count())
        # Use a more descriptive thread name prefix
        self._executor = ThreadPoolExecutor(
            max_workers=resolved_workers, thread_name_prefix="TranscriptionWorker"
        )
        # Store mapping from Future to Task for context retrieval on completion/error
        self._futures: Dict[Future[Any], Task[Any, Result[Any]]] = {}
        self._lock = threading.RLock()
        self._shutdown = False
        logging.info(f"WorkDistributor initialized with {resolved_workers} workers.")

    def submit(self, task: Task[TaskParams, TaskResultType]) -> Future[TaskResultType]:
        """
        Submits a Task for asynchronous execution.

        Args:
            task: The Task object to execute. Its `func` must return a `Result`.

        Returns:
            A Future representing the pending result (`TaskResultType`, which is a `Result`).

        Raises:
            RuntimeError: If the distributor has been shut down.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("WorkDistributor has been shut down.")

            # Submit the wrapper function which handles metrics and error catching
            future: Future[TaskResultType] = self._executor.submit(
                self._execute_task_with_wrapper, task
            )
            # Store the original task associated with the future
            self._futures[future] = task
            return future

    def _execute_task_with_wrapper(
        self, task: Task[TaskParams, TaskResultType]
    ) -> TaskResultType:
        """
        Internal wrapper: executes task, measures, handles unexpected errors.

        Ensures that the function always returns a `Result`, even if the underlying
        task function fails unexpectedly or doesn't return a `Result`.

        Args:
            task: The task to execute.

        Returns:
            A `Result` object (TaskResultType).
        """
        operation_name = task.context.get("operation", f"task_{task.func.__name__}")
        measurement_ctx = task.context
        # Add task priority to context if not present
        measurement_ctx.setdefault("priority", task.priority.name)

        with measure_execution(operation_name, measurement_ctx):
            try:
                # Execute the task function, which should return a Result
                result = task.execute()

                # Defensive check: Ensure the task function returned a Result
                if not isinstance(result, Result):
                    logging.warning(
                        f"Task function {task.func.__name__} did not return a Result object. "
                        f"Wrapping raw return value {type(result)} as success. This may hide errors."
                    )
                    # Attempt to cast the raw result to the expected success value type.
                    # This relies on TaskResultType being Result[Something].
                    # We create a success Result containing the raw value.
                    # This is risky if the raw value indicates an error state.
                    # A better approach is strict enforcement of Result return type.
                    return Result.success(result)  # type: ignore[return-value] # Let mypy infer R from result

                # Return the Result produced by the task function
                return result

            except Exception as e:
                # Catch unexpected exceptions *during* task execution
                logging.exception(
                    f"Unexpected error executing task '{operation_name}'. Context: {task.context}",
                    exc_info=e,
                )
                # Create a failure Result
                # We need to return TaskResultType, which is bound=Result[Any].
                # So, we create a Result[Any] failure.
                return Result.failure(  # type: ignore[return-value]
                    Error.create(
                        message=f"Task execution wrapper caught exception: {e}",
                        code="TASK_WRAPPER_UNEXPECTED_ERROR",
                        category=ErrorCategory.UNEXPECTED,
                        severity=ErrorSeverity.ERROR,
                        context=task.context,
                        exception=e,
                    )
                )

    def map(
        self,
        func: Callable[[TaskParams], TaskResultType],
        params_list: Iterable[TaskParams],
        priority: TaskPriority = TaskPriority.NORMAL,
        base_context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[TaskParams, TaskResultType]]:
        """
        Applies `func` (returning a Result) to each item in `params_list` concurrently.

        Yields `(params, result)` tuples as tasks complete. Order is not guaranteed.

        Args:
            func: Function accepting `TaskParams`, returning `TaskResultType` (a `Result`).
            params_list: Iterable of parameters to process.
            priority: Priority for these tasks.
            base_context: Optional base context dictionary for all tasks.

        Yields:
            Tuples of `(TaskParams, TaskResultType)` as tasks complete.

        Raises:
            RuntimeError: If the distributor is shut down.
            KeyboardInterrupt: If interrupted by the user during iteration.
        """
        if not params_list:
            logging.debug("WorkDistributor.map called with empty params_list.")
            return iter([])  # Return an empty iterator immediately

        # Use a list to allow getting length if needed, but still works with iterators
        params_sequence: Sequence[TaskParams] = (
            list(params_list) if not isinstance(params_list, Sequence) else params_list
        )
        total_tasks = len(params_sequence)
        logging.debug(f"WorkDistributor.map preparing to submit {total_tasks} tasks.")

        futures_map: Dict[Future[TaskResultType], TaskParams] = {}

        with self._lock:
            if self._shutdown:
                raise RuntimeError("WorkDistributor has been shut down.")

            for params in params_sequence:
                task_context = dict(base_context or {})
                # Add minimal context, avoid large objects
                task_context.setdefault("item_repr", repr(params)[:100])

                task = Task(
                    priority=priority, func=func, params=params, context=task_context
                )
                try:
                    future = self.submit(task)
                    futures_map[future] = params
                except RuntimeError:
                    logging.warning(
                        "WorkDistributor shut down during task submission in map."
                    )
                    break  # Stop submitting if shutdown occurs mid-loop

        submitted_count = len(futures_map)
        logging.info(f"Submitted {submitted_count}/{total_tasks} tasks via map.")

        # Yield results as they complete using as_completed
        completed_count = 0
        try:
            for future in as_completed(futures_map):
                params = futures_map[future]
                try:
                    # Retrieve the result (which should be a Result object)
                    result: TaskResultType = future.result()
                    completed_count += 1
                    logging.debug(
                        f"Task completed ({completed_count}/{submitted_count}). Params: {repr(params)[:100]}"
                    )
                    yield params, result
                except Exception as e:
                    # This catches exceptions *retrieving* the result from the future
                    # (e.g., if the task was cancelled or raised an unexpected error
                    # not caught by the wrapper).
                    completed_count += 1
                    logging.exception(
                        f"Unexpected error retrieving result for task ({completed_count}/{submitted_count}). Params: {repr(params)[:100]}",
                        exc_info=e,
                    )
                    # Yield a failure Result
                    yield params, Result.failure(  # type: ignore[misc] # R is Any here
                        Error.create(
                            message=f"Failed to retrieve task result: {e}",
                            code="FUTURE_RESULT_RETRIEVAL_ERROR",
                            category=ErrorCategory.UNEXPECTED,
                            severity=ErrorSeverity.ERROR,
                            context={"params_repr": repr(params)[:100]},
                            exception=e,
                        )
                    )
                finally:
                    # Clean up future from internal tracking *after* processing
                    with self._lock:
                        self._futures.pop(future, None)

        except KeyboardInterrupt:
            logging.warning("Map operation interrupted by user. Initiating shutdown...")
            # Don't wait for running tasks, cancel if possible
            self.shutdown(wait=False, cancel_futures=True)
            raise  # Re-raise interrupt to signal termination

        logging.debug(
            f"WorkDistributor.map finished processing {completed_count} tasks."
        )

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Shuts down the thread pool executor gracefully or forcefully.

        Args:
            wait: If True (default), waits for currently running tasks to complete.
                  If False, attempts to shut down immediately.
            cancel_futures: If True (and Python 3.9+), attempts to cancel
                            pending (not yet started) futures. Only effective
                            if `wait` is False.
        """
        should_cancel = sys.version_info >= (3, 9) and cancel_futures and not wait

        with self._lock:
            if self._shutdown:
                logging.debug("WorkDistributor already shut down.")
                return
            self._shutdown = True
            logging.info(
                f"Shutting down WorkDistributor (wait={wait}, cancel={should_cancel})..."
            )

            # Attempt cancellation first if requested
            if should_cancel:
                cancelled_count = 0
                # Iterate over a copy of keys as future.cancel() might modify the dict via callbacks
                future_list = list(self._futures.keys())
                for future in future_list:
                    if future.cancel():
                        cancelled_count += 1
                logging.debug(f"Attempted to cancel {cancelled_count} pending futures.")

        # Perform the actual executor shutdown
        # Note: shutdown() itself handles joining threads.
        self._executor.shutdown(wait=wait)

        # Clear remaining futures map after shutdown
        with self._lock:
            self._futures.clear()

        logging.info("WorkDistributor shut down complete.")


# =============================================================================
# 🔊 Core Transcription Logic
# =============================================================================


def check_dependencies() -> Result[None]:
    """
    Verifies that essential dependencies (Whisper, FFmpeg lib, FFmpeg exe) are available.

    Returns:
        Result[None]: `Result.success(None)` if all dependencies are met,
                      `Result.failure(Error)` otherwise.
    """
    if whisper is None:
        return Result.failure(
            Error.create(
                message="Whisper library not found. Please install with: pip install openai-whisper",
                code="MISSING_DEPENDENCY_WHISPER",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.FATAL,
                context={"missing_package": "openai-whisper"},
            )
        )

    if ffmpeg is None:
        return Result.failure(
            Error.create(
                message="ffmpeg-python library not found. Please install with: pip install ffmpeg-python",
                code="MISSING_DEPENDENCY_FFMPEG_PY",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.FATAL,
                context={"missing_package": "ffmpeg-python"},
            )
        )

    # Check if FFmpeg executable is available and runnable
    try:
        # Use ffmpeg-python's way of finding the command if possible, else default 'ffmpeg'
        cmd = getattr(ffmpeg, "cmd", "ffmpeg")
        process = subprocess.run(
            [cmd, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding="utf-8",
            errors="replace",  # Handle potential decoding errors
        )
        logging.debug(
            f"FFmpeg version check successful: {process.stdout.splitlines()[0]}"
        )
    except FileNotFoundError:
        return Result.failure(
            Error.create(
                message="FFmpeg executable not found in system PATH. Please install FFmpeg.",
                code="MISSING_EXECUTABLE_FFMPEG",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.FATAL,
                context={"executable_checked": getattr(ffmpeg, "cmd", "ffmpeg")},
            )
        )
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr or "No stderr captured"
        return Result.failure(
            Error.create(
                message=f"FFmpeg executable found but failed execution: {stderr_msg}",
                code="INVALID_EXECUTABLE_FFMPEG",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.FATAL,
                context={
                    "executable": getattr(ffmpeg, "cmd", "ffmpeg"),
                    "stderr": stderr_msg,
                },
                exception=e,
            )
        )
    except Exception as e:
        # Catch other potential errors during subprocess execution
        return Result.failure(
            Error.create(
                message=f"Unexpected error checking FFmpeg executable: {e}",
                code="FFMPEG_CHECK_UNEXPECTED_ERROR",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.FATAL,
                context={"executable": getattr(ffmpeg, "cmd", "ffmpeg")},
                exception=e,
            )
        )

    return Result.success(None)


def is_media_file(filepath: str) -> bool:
    """
    Checks if a file is a supported audio or video file based on its extension.

    Args:
        filepath: Path to the file.

    Returns:
        True if the file extension is in `AUDIO_EXTENSIONS` or `VIDEO_EXTENSIONS`.
    """
    # Use pathlib for more robust path manipulation
    return Path(filepath).suffix.lower() in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def validate_model_size(model_size: str) -> Result[ModelSize]:
    """
    Validates if the provided model size string is a valid `ModelSize`.

    Args:
        model_size: The model size string to validate (case-insensitive).

    Returns:
        `Result.success(validated_model_size)` if valid, `Result.failure(Error)` otherwise.
    """
    normalized_size = model_size.lower()
    if normalized_size in VALID_MODEL_SIZES:
        # Cast is safe here due to the check
        return Result.success(cast(ModelSize, normalized_size))
    else:
        valid_options = ", ".join(sorted(VALID_MODEL_SIZES))
        return Result.failure(
            Error.create(
                message=f"Invalid model size '{model_size}'. Valid options are: {valid_options}",
                code="INVALID_MODEL_SIZE",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.ERROR,
                context={"provided": model_size, "valid_options": valid_options},
            )
        )


def extract_audio(
    input_path: str, output_path: str, sample_rate: int = 16000
) -> Result[str]:
    """
    Extracts audio from a media file to WAV format using FFmpeg.

    Outputs a mono WAV file at the specified sample rate (default 16kHz),
    optimized for Whisper. Overwrites the output file if it exists.

    Args:
        input_path: Path to the input media file (audio or video).
        output_path: Path to save the extracted WAV audio file.
        sample_rate: Target audio sample rate in Hz (default: 16000).

    Returns:
        `Result.success(output_path)` on success, `Result.failure(Error)` otherwise.
    """
    if ffmpeg is None:
        # This check should ideally be done once at startup, but included for robustness
        return Result.failure(
            Error.create(
                message="Cannot extract audio: ffmpeg-python library not installed.",
                code="MISSING_DEPENDENCY_FFMPEG_PY",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.ERROR,
            )
        )

    measurement_ctx = {"input": input_path, "output": output_path, "rate": sample_rate}
    with measure_execution("extract_audio", measurement_ctx):
        logging.info(f"Extracting audio from '{input_path}' to '{output_path}'...")
        try:
            # Build FFmpeg command using ffmpeg-python
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream,
                output_path,
                format="wav",  # Output format
                acodec="pcm_s16le",  # Standard WAV codec
                ac=1,  # Mono channel
                ar=str(sample_rate),  # Sample rate
            )
            # Run FFmpeg, capturing output, overwriting existing file, suppressing console output
            stdout, stderr = ffmpeg.run(
                stream,
                cmd=getattr(ffmpeg, "cmd", "ffmpeg"),  # Use configured cmd or default
                capture_stdout=True,
                capture_stderr=True,
                overwrite_output=True,
                quiet=True,  # Suppress FFmpeg console logs
            )

            logging.info(f"Audio successfully extracted to '{output_path}'.")
            # Log FFmpeg output if needed for debugging
            # logging.debug(f"FFmpeg stdout:\n{stdout.decode(errors='replace')}")
            # logging.debug(f"FFmpeg stderr:\n{stderr.decode(errors='replace')}")
            return Result.success(output_path)

        except ffmpeg.Error as e:
            # Handle errors reported by ffmpeg-python
            stderr_msg = (
                e.stderr.decode("utf-8", errors="replace").strip()
                if e.stderr
                else "No stderr captured"
            )
            logging.error(f"FFmpeg audio extraction failed: {stderr_msg}")
            return Result.failure(
                Error.create(
                    message=f"Audio extraction failed: {stderr_msg}",
                    code="AUDIO_EXTRACTION_FAILED",
                    category=ErrorCategory.EXTERNAL,
                    severity=ErrorSeverity.ERROR,
                    context={
                        "input": input_path,
                        "output": output_path,
                        "stderr": stderr_msg,
                    },
                    exception=e,
                )
            )
        except Exception as e:
            # Catch any other unexpected errors during the process
            logging.exception(
                f"Unexpected error during audio extraction for {input_path}"
            )
            return Result.failure(
                Error.create(
                    message=f"Unexpected audio extraction error: {e}",
                    code="AUDIO_EXTRACTION_UNEXPECTED_ERROR",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                    context={"input": input_path, "output": output_path},
                    exception=e,
                )
            )


def format_timestamp(seconds: float) -> str:
    """
    Formats seconds into a human-readable HH:MM:SS string.

    Handles non-negative inputs. Rounds down to the nearest second.

    Args:
        seconds: The number of seconds (float, expected non-negative).

    Returns:
        A string representation in `H:MM:SS` format (e.g., "0:01:10", "1:01:01").
    """
    if seconds < 0:
        logging.warning(
            f"Received negative timestamp ({seconds}), formatting as 0:00:00."
        )
        seconds = 0
    # Use timedelta for robust formatting, converting to int first
    delta = timedelta(seconds=int(seconds))
    # str(delta) gives "H:MM:SS" or "D days, H:MM:SS" - we only want H:MM:SS part
    return str(delta).split(",")[-1].strip()


def transcribe_audio(
    audio_path: str, model_size: ModelSize
) -> Result[TranscriptionResult]:
    """
    Transcribes an audio file using the specified Whisper model on the CPU.

    Loads the model if not already cached by Whisper. Forces CPU execution.

    Args:
        audio_path: Path to the WAV audio file (ideally 16kHz mono).
        model_size: The size of the Whisper model to use (`tiny`, `base`, etc.).

    Returns:
        `Result.success(transcription_data)` containing the `WhisperResult` dict,
        or `Result.failure(Error)` if transcription fails.
    """
    if whisper is None or Whisper is None:
        return Result.failure(
            Error.create(
                message="Cannot transcribe: Whisper library not installed or loaded.",
                code="MISSING_DEPENDENCY_WHISPER",
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.ERROR,
            )
        )

    measurement_ctx = {"audio_path": audio_path, "model": model_size}
    with measure_execution("transcribe_audio", measurement_ctx):
        try:
            # Load the model - Whisper handles caching internally
            # Force CPU usage via device="cpu"
            logging.info(f"Loading Whisper model '{model_size}' (forced CPU)...")
            model: Whisper = whisper.load_model(model_size, device="cpu")
            logging.info(
                f"Model '{model_size}' loaded. Starting transcription for '{audio_path}'..."
            )

            # Perform transcription
            # fp16=False is essential for CPU execution.
            # language=None enables auto-detection.
            # word_timestamps=False by default, can be enabled if needed.
            result_raw: Dict[str, Any] = model.transcribe(
                audio_path, fp16=False, language=None, word_timestamps=False
            )

            logging.info(f"Transcription complete for '{audio_path}'.")

            # Basic validation of the returned structure before casting
            if (
                isinstance(result_raw, dict)
                and isinstance(result_raw.get("text"), str)
                and isinstance(result_raw.get("segments"), list)
                and isinstance(result_raw.get("language"), str)
            ):
                # Further validation of segment structure could be added here if needed
                # Cast to the specific WhisperResult TypedDict for type safety downstream
                return Result.success(cast(WhisperResult, result_raw))
            else:
                logging.error(
                    f"Whisper returned unexpected result format: {type(result_raw)}. Keys: {result_raw.keys() if isinstance(result_raw, dict) else 'N/A'}"
                )
                return Result.failure(
                    Error.create(
                        message="Whisper transcription returned an unexpected data structure.",
                        code="TRANSCRIPTION_INVALID_FORMAT",
                        category=ErrorCategory.EXTERNAL,
                        severity=ErrorSeverity.ERROR,
                        context={
                            "audio_path": audio_path,
                            "model": model_size,
                            "result_type": str(type(result_raw)),
                            "result_keys": (
                                list(result_raw.keys())
                                if isinstance(result_raw, dict)
                                else None
                            ),
                        },
                    )
                )

        except Exception as e:
            # Catch potential errors during model loading or transcription execution
            logging.exception(f"Error during Whisper transcription for '{audio_path}'")
            return Result.failure(
                Error.create(
                    message=f"Whisper transcription failed: {e}",
                    code="TRANSCRIPTION_FAILED",
                    category=ErrorCategory.EXTERNAL,
                    severity=ErrorSeverity.ERROR,
                    context={"audio_path": audio_path, "model": model_size},
                    exception=e,
                )
            )


def generate_formatted_transcript(result: TranscriptionResult) -> str:
    """
    Formats the raw Whisper transcription result into human-readable text with timestamps.

    Args:
        result: The `WhisperResult` dictionary from `transcribe_audio`.

    Returns:
        A formatted string containing the transcript, with each segment prefixed
        by its start time (e.g., "[0:00:15] Text of the segment."). Returns
        an empty string if the result contains no valid segments or text.
    """
    with measure_execution("format_transcript"):
        transcript_lines: List[str] = []
        # Use .get() for safer access in case the structure is malformed
        segments: List[WhisperSegment] = result.get("segments", [])

        if not segments:
            # Fallback if segments list is empty or missing
            full_text = result.get("text", "").strip()
            if full_text:
                logging.warning(
                    "Transcription segments missing or empty, using full text without timestamps."
                )
                # Return just the text if no segments are available
                return full_text
            else:
                logging.warning("No text or segments found in transcription result.")
                return ""  # Return empty string if truly no content

        # Process segments if available
        for i, seg in enumerate(segments):
            # Validate segment structure before accessing keys
            if (
                isinstance(seg, dict)
                and isinstance(seg.get("start"), (int, float))
                and isinstance(seg.get("text"), str)
            ):
                start_time: float = seg["start"]
                text: str = seg["text"].strip()
                # Only add line if text is not empty
                if text:
                    timestamp_str = format_timestamp(start_time)
                    transcript_lines.append(f"[{timestamp_str}] {text}")
            else:
                logging.warning(
                    f"Skipping invalid segment at index {i}: {repr(seg)[:100]}"
                )

        # Join lines with double newline for readability between segments
        return "\n\n".join(transcript_lines)


def save_transcript(
    transcript_text: str, output_path: str, encoding: str = "utf-8"
) -> Result[str]:
    """
    Saves the formatted transcript text to a specified file path.

    Creates the output directory if it doesn't exist. Handles potential file system errors.

    Args:
        transcript_text: The string content of the transcript to save.
        output_path: The full path to the file where the transcript will be saved.
        encoding: The text encoding to use (default: 'utf-8').

    Returns:
        `Result.success(output_path)` on successful save, `Result.failure(Error)` otherwise.
    """
    measurement_ctx = {"output_path": output_path, "encoding": encoding}
    with measure_execution("save_transcript", measurement_ctx):
        try:
            # Use pathlib for robust path operations
            output_file = Path(output_path)
            output_dir = output_file.parent

            # Ensure the output directory exists, creating it if necessary
            output_dir.mkdir(parents=True, exist_ok=True)

            # Write the transcript text to the file
            output_file.write_text(transcript_text, encoding=encoding, errors="replace")

            logging.info(f"Transcript successfully saved to: {output_path}")
            return Result.success(output_path)

        except OSError as e:
            # Catch file system related errors (permissions, disk full, invalid path etc.)
            logging.exception(f"OS error saving transcript to {output_path}")
            return Result.failure(
                Error.create(
                    message=f"Failed to save transcript file: {e.strerror}",
                    code="FILE_WRITE_OS_ERROR",
                    category=ErrorCategory.RESOURCE,  # Changed category to RESOURCE
                    severity=ErrorSeverity.ERROR,
                    context={"output_path": output_path, "errno": e.errno},
                    exception=e,
                )
            )
        except Exception as e:
            # Catch any other unexpected errors during file writing
            logging.exception(f"Unexpected error saving transcript to {output_path}")
            return Result.failure(
                Error.create(
                    message=f"Unexpected error saving transcript: {e}",
                    code="FILE_WRITE_UNEXPECTED_ERROR",
                    category=ErrorCategory.UNEXPECTED,
                    severity=ErrorSeverity.ERROR,
                    context={"output_path": output_path},
                    exception=e,
                )
            )


# =============================================================================
# 🔄 File Processing Pipeline
# =============================================================================


@dataclass(frozen=True)
class PreparedAudio:
    """
    Holds the path to the audio file ready for transcription and its temporary status.

    Attributes:
        path: Path to the audio file (WAV format, ideally 16kHz mono).
        is_temporary: Boolean indicating if this file was created temporarily
                      (e.g., extracted from video) and should be cleaned up later.
    """

    path: str
    is_temporary: bool


def prepare_audio_for_transcription(input_path: str) -> Result[PreparedAudio]:
    """
    Ensures an audio file is ready for Whisper (WAV format). Extracts if necessary.

    Creates temporary WAV files for non-WAV audio or video inputs.

    Args:
        input_path: Path to the input media file (audio or video).

    Returns:
        `Result.success(PreparedAudio)` containing the path to the usable WAV audio
        file and its temporary status, or `Result.failure(Error)`.
    """
    with measure_execution("prepare_audio", {"input": input_path}):
        input_file = Path(input_path)
        ext_lower = input_file.suffix.lower()

        if ext_lower == ".wav":
            # Assume WAV is likely okay, proceed directly.
            # Whisper handles sample rate differences to some extent.
            logging.debug(f"Input '{input_path}' is WAV, using directly.")
            return Result.success(PreparedAudio(path=input_path, is_temporary=False))

        elif ext_lower in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
            action = (
                "Converting"
                if ext_lower in AUDIO_EXTENSIONS
                else "Extracting audio from"
            )
            logging.info(f"{action} '{input_path}' to temporary WAV file...")
            temp_file_result = _create_temp_wav_file(input_path)

            if temp_file_result.is_failure:
                return Result.failure(temp_file_result.error)  # Propagate error

            tmp_wav_path = temp_file_result.unwrap()

            # Perform the extraction/conversion
            extract_result = extract_audio(input_path, tmp_wav_path)

            if extract_result.is_failure:
                # Clean up the failed temporary file attempt
                _safe_remove_file(tmp_wav_path, "temporary WAV after failed extraction")
                # Ensure the failure result is correctly typed for PreparedAudio
                return Result.failure(extract_result.error)

            # Extraction succeeded, return the path to the temporary file
            return Result.success(PreparedAudio(path=tmp_wav_path, is_temporary=True))

        else:
            # Unsupported file type
            supported_types = sorted(list(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))
            return Result.failure(
                Error.create(
                    message=f"Unsupported file type: '{ext_lower}'. Supported: {', '.join(supported_types)}",
                    code="UNSUPPORTED_FILE_TYPE",
                    category=ErrorCategory.VALIDATION,
                    severity=ErrorSeverity.ERROR,
                    context={"input": input_path, "extension": ext_lower},
                )
            )


def _create_temp_wav_file(source_path: str) -> Result[str]:
    """Helper to create a temporary file path with .wav suffix."""
    try:
        # Create a temporary file, get its path, then close/delete it
        # so ffmpeg can create it without permission issues.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_wav_path = tmp_file.name
        # Ensure the file is removed initially if NamedTemporaryFile didn't delete on close
        _safe_remove_file(tmp_wav_path, "initial temporary file placeholder")
        return Result.success(tmp_wav_path)
    except Exception as e:
        return Result.failure(
            Error.create(
                message=f"Failed creating temporary file for '{source_path}': {e}",
                code="TEMP_FILE_CREATION_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.ERROR,
                context={"source": source_path},
                exception=e,
            )
        )


def _safe_remove_file(filepath: str, context_msg: str = "") -> None:
    """Safely removes a file, logging warnings on failure."""
    try:
        if os.path.exists(filepath):  # Check existence before removal
            os.remove(filepath)
            logging.debug(f"Removed {context_msg}: {filepath}")
    except OSError as e:
        logging.warning(f"Failed to remove {context_msg} '{filepath}': {e.strerror}")
    except Exception as e:
        logging.warning(f"Unexpected error removing {context_msg} '{filepath}': {e}")


def cleanup_temporary_file(prepared_audio: PreparedAudio) -> None:
    """
    Safely removes the audio file if it was marked as temporary.

    Logs a warning if removal fails but does not raise an error.

    Args:
        prepared_audio: The `PreparedAudio` object containing file path and status.
    """
    if prepared_audio.is_temporary:
        _safe_remove_file(prepared_audio.path, "temporary audio file")


@dataclass(frozen=True)
class ProcessFileParams:
    """Parameters required for processing a single media file."""

    input_path: str
    output_path: str
    model_size: ModelSize


def process_single_file(params: ProcessFileParams) -> Result[str]:
    """
    Executes the full transcription pipeline for one media file.

    Orchestrates audio preparation, transcription, formatting, saving, and cleanup.
    Uses the `Result` monad's `flat_map` for cleaner pipeline chaining.

    Args:
        params: A `ProcessFileParams` object with input/output paths and model size.

    Returns:
        `Result.success(output_path)` on successful completion, or `Result.failure(Error)`.
    """
    measurement_ctx = {
        "input": params.input_path,
        "output": params.output_path,
        "model": params.model_size,
    }
    with measure_execution("process_single_file", measurement_ctx):
        logging.info(f"Starting processing for: {params.input_path}")

        # --- Input Validation ---
        input_file = Path(params.input_path)
        if not input_file.exists():
            return Result.failure(
                Error.create(
                    f"Input file not found: {params.input_path}",
                    "FILE_NOT_FOUND",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"path": params.input_path},
                )
            )
        if not input_file.is_file():
            return Result.failure(
                Error.create(
                    f"Input path is not a file: {params.input_path}",
                    "PATH_NOT_FILE",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"path": params.input_path},
                )
            )

        # --- Pipeline Execution ---
        prepared_audio_container: List[Optional[PreparedAudio]] = [
            None
        ]  # Use list for mutable closure

        try:
            # Chain operations using flat_map for cleaner error propagation
            final_result: Result[str] = (
                prepare_audio_for_transcription(params.input_path)
                .flat_map(
                    lambda pa: (
                        # Store prepared audio info for finally block cleanup
                        prepared_audio_container.__setitem__(0, pa),
                        transcribe_audio(pa.path, params.model_size),
                    )[
                        1
                    ]  # Return only the transcription result
                )
                .map(generate_formatted_transcript)
                .flat_map(lambda text: save_transcript(text, params.output_path))
            )

            # Log final outcome based on the result
            if final_result.is_success:
                logging.info(
                    f"Successfully processed '{params.input_path}' -> '{final_result.unwrap()}'"
                )
            else:
                # Error should already be logged by the failing step, but log summary here
                error = final_result.error
                # Check error is not None before accessing attributes
                if error:
                    logging.error(
                        f"Failed to process '{params.input_path}': [{error.code}] {error.message}"
                    )
                    # Optionally log traceback for detailed debugging if available and level allows
                    if error.trace and logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(
                            f"Traceback for {params.input_path}:\n{error.trace}"
                        )
                else:
                    logging.error(
                        f"Failed to process '{params.input_path}' with unknown error."
                    )

            return final_result

        except Exception as e:
            # Catch truly unexpected errors in the pipeline orchestration itself
            logging.exception(f"Unexpected pipeline error for {params.input_path}")
            return Result.failure(
                Error.create(
                    f"Unexpected pipeline error: {e}",
                    "PIPELINE_UNEXPECTED_ERROR",
                    ErrorCategory.UNEXPECTED,
                    Severity=ErrorSeverity.FATAL,  # Pipeline failure is serious
                    context={"input": params.input_path},
                    exception=e,
                )
            )
        finally:
            # --- Cleanup ---
            # Ensure temporary file is cleaned up regardless of success/failure
            prepared_audio = prepared_audio_container[0]
            if prepared_audio:
                cleanup_temporary_file(prepared_audio)


# =============================================================================
# 📂 Directory Processing Logic
# =============================================================================


def find_media_files(input_dir: str, recursive: bool = False) -> Iterator[str]:
    """
    Scans a directory for supported media files (audio/video).

    Args:
        input_dir: The directory path to scan.
        recursive: If True, scans subdirectories recursively.

    Yields:
        Absolute paths to found media files as strings.
    """
    with measure_execution(
        "find_media_files", {"dir": input_dir, "recursive": recursive}
    ):
        base_path = Path(input_dir)
        if not base_path.is_dir():
            logging.warning(f"Input path is not a directory: {input_dir}")
            return  # Yield nothing

        if recursive:
            # Use Path.rglob for efficient recursive search
            for item in base_path.rglob("*"):
                if item.is_file() and is_media_file(str(item)):
                    yield str(item.resolve())  # Yield absolute path string
        else:
            # Use Path.iterdir for non-recursive search
            try:
                for item in base_path.iterdir():
                    if item.is_file() and is_media_file(str(item)):
                        yield str(item.resolve())  # Yield absolute path string
            except OSError as e:
                logging.error(f"Error listing directory '{input_dir}': {e.strerror}")


def generate_output_path(
    input_file_path: str, input_base_dir: str, output_base_dir: str
) -> str:
    """
    Calculates the output transcript path, preserving relative directory structure.

    Example:
        input_file_path = /home/user/media/subdir/video.mp4
        input_base_dir  = /home/user/media
        output_base_dir = /home/user/transcripts
        Returns: /home/user/transcripts/subdir/video.txt

    Args:
        input_file_path: Absolute path to the input media file.
        input_base_dir: Absolute path to the base input directory.
        output_base_dir: Absolute path to the base output directory.

    Returns:
        Absolute path string for the corresponding output .txt file.
    """
    input_file = Path(input_file_path)
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)

    try:
        # Get path relative to the input base directory
        relative_path = input_file.relative_to(input_base)
    except ValueError:
        # Should not happen if input_file_path is within input_base_dir, but handle defensively
        logging.warning(
            f"Input file '{input_file}' not relative to base '{input_base}'. Using filename only."
        )
        relative_path = Path(input_file.name)

    # Change the suffix to .txt
    output_relative_path = relative_path.with_suffix(".txt")

    # Join with the output base directory and resolve to absolute path
    return str(output_base.joinpath(output_relative_path).resolve())


@dataclass(frozen=True)
class ProcessDirectoryParams:
    """Parameters for processing a directory of media files."""

    input_dir: str
    output_dir: str
    model_size: ModelSize
    recursive: bool
    worker_count: int


def process_directory(params: ProcessDirectoryParams) -> Result[List[str]]:
    """
    Processes all supported media files within a directory concurrently using WorkDistributor.

    Args:
        params: A `ProcessDirectoryParams` object containing configuration.

    Returns:
        `Result.success(list_of_output_paths)` containing paths of successfully
        created transcripts, or `Result.failure(Error)` if any file fails or
        a critical error occurs.
    """
    # Use dataclass fields directly for context
    measurement_ctx = params.__dict__
    with measure_execution("process_directory", measurement_ctx):
        input_path = Path(params.input_dir)
        output_path = Path(params.output_dir)

        # --- Input Validation ---
        if not input_path.exists():
            return Result.failure(
                Error.create(
                    f"Input directory not found: {params.input_dir}",
                    "DIR_NOT_FOUND",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"dir": params.input_dir},
                )
            )
        if not input_path.is_dir():
            return Result.failure(
                Error.create(
                    f"Input path is not a directory: {params.input_dir}",
                    "PATH_NOT_DIR",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"dir": params.input_dir},
                )
            )

        # --- Ensure Output Directory Exists ---
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return Result.failure(
                Error.create(
                    f"Cannot create output directory '{params.output_dir}': {e.strerror}",
                    "DIR_CREATE_FAILED",
                    ErrorCategory.RESOURCE,
                    ErrorSeverity.ERROR,
                    {"dir": params.output_dir},
                    exception=e,
                )
            )

        # --- Find Files ---
        # Collect all files first to get a total count for progress reporting
        media_files_to_process = list(
            find_media_files(params.input_dir, params.recursive)
        )
        total_files = len(media_files_to_process)

        if total_files == 0:
            logging.warning(f"No supported media files found in '{params.input_dir}'.")
            return Result.success([])  # Success, but nothing processed

        logging.info(
            f"Found {total_files} media files to process in '{params.input_dir}'."
        )

        # --- Prepare Task Parameters ---
        tasks_params: List[ProcessFileParams] = []
        for input_file in media_files_to_process:
            try:
                output_file = generate_output_path(
                    input_file, params.input_dir, params.output_dir
                )
                tasks_params.append(
                    ProcessFileParams(
                        input_path=input_file,
                        output_path=output_file,
                        model_size=params.model_size,
                    )
                )
            except Exception as e:
                # Handle errors during output path generation (e.g., path issues)
                logging.error(f"Failed to generate output path for '{input_file}': {e}")
                # Skip this file or return an error? Let's skip and log.
                # Could collect these errors and return a partial failure later.

        if not tasks_params:
            logging.error("No valid tasks could be created for directory processing.")
            return Result.failure(
                Error.create(
                    "Failed to prepare any tasks for processing.",
                    "TASK_PREPARATION_FAILED",
                    ErrorCategory.UNEXPECTED,
                    ErrorSeverity.ERROR,
                )
            )

        actual_tasks_count = len(tasks_params)
        if actual_tasks_count < total_files:
            logging.warning(
                f"Skipped {total_files - actual_tasks_count} files due to errors during task preparation."
            )

        # --- Execute Concurrently ---
        distributor = WorkDistributor(max_workers=params.worker_count)
        successful_output_paths: List[str] = []
        aggregated_errors: List[Error] = []
        processed_count = 0

        try:
            # Use distributor.map for concurrent execution
            # process_single_file matches the required signature: (P) -> Result[R]
            for task_p, result in distributor.map(
                process_single_file,
                tasks_params,  # List of ProcessFileParams
                base_context={"operation": "process_single_file"},
            ):
                processed_count += 1
                progress_percent = (processed_count / actual_tasks_count) * 100
                if result.is_success:
                    successful_output_paths.append(result.unwrap())
                    logging.info(
                        f"({processed_count}/{actual_tasks_count} - {progress_percent:.1f}%) SUCCESS: '{task_p.input_path}'"
                    )
                else:
                    # Ensure error is not None before accessing attributes
                    if result.error:
                        aggregated_errors.append(result.error)
                        logging.error(
                            f"({processed_count}/{actual_tasks_count} - {progress_percent:.1f}%) FAILURE: '{task_p.input_path}' - [{result.error.code}] {result.error.message}"
                        )
                        # Optionally log traceback for debugging
                        if result.error.trace and logging.getLogger().isEnabledFor(
                            logging.DEBUG
                        ):
                            logging.debug(
                                f"Traceback for {task_p.input_path}:\n{result.error.trace}"
                            )
                    else:
                        # This case should be rare due to Result structure
                        logging.error(
                            f"({processed_count}/{actual_tasks_count} - {progress_percent:.1f}%) FAILURE: '{task_p.input_path}' - Unknown error"
                        )
                        aggregated_errors.append(
                            Error.create(
                                "Unknown processing error",
                                "UNKNOWN_PROCESSING_FAILURE",
                                ErrorCategory.UNEXPECTED,
                                ErrorSeverity.ERROR,
                            )
                        )

        except KeyboardInterrupt:
            logging.warning("Directory processing interrupted by user.")
            # distributor.shutdown is handled by map's finally block
            return Result.failure(
                Error.create(
                    "Processing interrupted by user.",
                    "USER_INTERRUPT",
                    ErrorCategory.UNEXPECTED,
                    ErrorSeverity.WARNING,
                )
            )
        except Exception as e:
            # Catch unexpected errors during the mapping/distribution process
            logging.exception(
                "Unexpected error during directory processing orchestration."
            )
            distributor.shutdown(wait=False, cancel_futures=True)  # Attempt cleanup
            return Result.failure(
                Error.create(
                    f"Orchestration error: {e}",
                    "DIR_PROCESS_ORCHESTRATION_ERROR",
                    ErrorCategory.UNEXPECTED,
                    ErrorSeverity.FATAL,
                    exception=e,
                )
            )
        finally:
            # Ensure distributor is shut down even if map exits early or normally
            distributor.shutdown(wait=True)  # Wait for any running tasks to finish

        # --- Final Outcome ---
        failure_count = len(aggregated_errors)
        success_count = len(successful_output_paths)
        logging.info(
            f"Directory processing complete. Total Attempted: {actual_tasks_count}, Succeeded: {success_count}, Failed: {failure_count}"
        )

        if failure_count > 0:
            # Return failure if any file failed
            summary_message = (
                f"{failure_count}/{actual_tasks_count} files failed to process."
            )
            first_error = aggregated_errors[0]  # Get the first error for context

            return Result.failure(
                Error.create(
                    message=summary_message,
                    code="DIR_PROCESSING_PARTIAL_FAILURE",
                    category=first_error.category,  # Use category of first error
                    severity=ErrorSeverity.ERROR,  # Indicate overall process had errors
                    context={
                        "total_attempted": actual_tasks_count,
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "first_error_code": first_error.code,
                        "all_error_codes": [e.code for e in aggregated_errors],
                        # Optionally include full errors if needed, but can be large
                        # "first_error_details": first_error,
                    },
                )
            )
        else:
            # All attempted files succeeded
            return Result.success(successful_output_paths)


# =============================================================================
# ⌨️ Command Line Interface and Interactive Mode
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments using argparse.

    Defines arguments for input/output paths, model size, recursion, workers,
    stats display, and logging level. Includes default values and help messages.

    Returns:
        An `argparse.Namespace` object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Eidosian Offline Audio/Video Transcription using Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python av_to_text.py -i input.mp4 -o output.txt -m small",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,  # Default None triggers interactive mode if output is also None
        help="Path to the input audio/video file or directory.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,  # Default None triggers interactive mode if input is also None
        help="Path for the output transcript file or directory.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=_DEFAULT_MODEL_SIZE,
        choices=sorted(list(VALID_MODEL_SIZES)),  # Enforce valid choices
        help="Whisper model size to use.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process directories recursively (only applies if input is a directory).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=max(1, cpu_count() // 2),  # Default to half CPU cores, minimum 1
        help="Number of worker threads for parallel directory processing.",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show performance statistics after processing completes.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging output level.",
    )

    args = parser.parse_args()

    # Post-parse validation
    if args.workers < 1:
        parser.error("Number of workers must be at least 1.")

    # If input is provided, output must also be provided for non-interactive mode
    if args.input is not None and args.output is None:
        parser.error("Argument --output/-o is required when --input/-i is provided.")
    # If output is provided, input must also be provided
    if args.output is not None and args.input is None:
        parser.error("Argument --input/-i is required when --output/-o is provided.")

    return args


def configure_logging(level: str = "INFO") -> None:
    """
    Sets up basic logging configuration for the script.

    Args:
        level: The desired logging level as a string (e.g., "INFO", "DEBUG").
               Defaults to "INFO".
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    # Use a slightly more detailed format
    log_format = "%(asctime)s [%(levelname)-8s] [%(threadName)-15s] %(message)s"
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        # Force logging to stderr? Or allow default stdout/stderr handling?
        # stream=sys.stderr # Uncomment to force all logs to stderr
    )
    # Suppress overly verbose logs from dependencies if needed
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)  # Often noisy on import


def prompt_user(
    prompt_text: str, validation_func: Optional[Callable[[str], bool]] = None
) -> str:
    """
    Helper function to prompt the user for input and optionally validate it.

    Handles `EOFError` and `KeyboardInterrupt` gracefully by exiting.

    Args:
        prompt_text: The text to display to the user (e.g., "Enter path").
        validation_func: An optional function that takes the user's input string
                         and returns `True` if valid, `False` otherwise. If `False`,
                         the user is re-prompted.

    Returns:
        The validated user input string.

    Raises:
        SystemExit: If the user cancels via EOF (Ctrl+D) or KeyboardInterrupt (Ctrl+C).
    """
    while True:
        try:
            user_input = input(f"{prompt_text}: ").strip()
            if not user_input:  # Handle empty input if needed, re-prompt by default
                print("Input cannot be empty. Please try again.")
                continue
            if validation_func is None or validation_func(user_input):
                return user_input
            else:
                # Validation function should ideally print its own error message
                # via the print() in the lambda, but add a generic fallback.
                # print("Invalid input. Please try again.") # Redundant if lambda prints
                pass  # Assume validation_func printed the error
        except EOFError:
            print("\n🚫 Operation cancelled by EOF.")
            sys.exit(1)  # Standard exit code for cancellation
        except KeyboardInterrupt:
            print("\n🚫 Operation cancelled by user.")
            sys.exit(130)  # Standard exit code for Ctrl+C


def interactive_mode() -> Tuple[str, str, ModelSize, bool, int]:
    """
    Guides the user through an interactive session to gather processing parameters.

    Returns:
        A tuple containing:
        `(input_path: str, output_path: str, model_size: ModelSize, recursive: bool, worker_count: int)`.

    Raises:
        SystemExit: If the user cancels the operation at the confirmation step.
    """
    print("\n=== Eidosian Audio/Video Transcription (Interactive Mode) ===")

    # 1. Get Input Path (validated for existence)
    input_path = prompt_user(
        "Enter path to input file or directory",
        lambda p: Path(p).exists()
        or print(f"❌ Error: Path '{p}' does not exist.")
        is None,  # Use print in lambda for inline error
    )
    input_path_abs = str(Path(input_path).resolve())  # Store absolute path
    is_input_dir = Path(input_path_abs).is_dir()

    # 2. Get Recursive Option (only if input is directory)
    recursive = False
    if is_input_dir:
        recursive_choice = prompt_user(
            "Process directory recursively? (y/n)",
            lambda c: c.lower() in ["y", "yes", "n", "no"]
            or print("❌ Enter 'y' or 'n'.") is None,
        )
        recursive = recursive_choice.lower().startswith("y")

    # 3. Get Output Path
    output_prompt = (
        "Enter path for output directory"
        if is_input_dir
        else "Enter path for output transcript file (e.g., transcript.txt)"
    )
    output_path = prompt_user(output_prompt)
    output_path_abs = str(Path(output_path).resolve())  # Store absolute path

    # Basic validation/warning for output based on input type
    output_is_dir = Path(
        output_path_abs
    ).is_dir()  # Check if it *currently* exists as dir
    output_looks_like_file = Path(output_path_abs).suffix != ""

    if is_input_dir and output_looks_like_file and not output_is_dir:
        print(
            f"⚠️ Warning: Output path '{output_path}' looks like a file, but input is a directory. Output will be treated as a directory path."
        )
    elif not is_input_dir and not output_looks_like_file:
        print(
            f"⚠️ Warning: Output path '{output_path}' does not look like a file (no extension). Ensure this is intended."
        )
    elif not is_input_dir and not output_path_abs.lower().endswith(".txt"):
        print(f"⚠️ Warning: Output file path '{output_path}' does not end with .txt.")

    # 4. Get Worker Count (only if input is directory)
    worker_count = max(1, cpu_count() // 2)  # Default
    if is_input_dir:
        worker_input = prompt_user(
            f"Number of worker threads [{worker_count}]",
            lambda w: w == ""
            or (w.isdigit() and int(w) >= 1)
            or print("❌ Must be a positive integer.") is None,
        )
        worker_count = int(worker_input) if worker_input else worker_count

    # 5. Get Model Size
    print("\nSelect Whisper model size:")
    model_options = sorted(list(VALID_MODEL_SIZES))
    for i, size in enumerate(model_options, 1):
        print(f"  {i}. {size}")
    model_choice_input = prompt_user(
        f"Enter choice number (1-{len(model_options)}) [{model_options.index(_DEFAULT_MODEL_SIZE)+1}]",
        lambda c: c == ""
        or (c.isdigit() and 1 <= int(c) <= len(model_options))
        or print(f"❌ Enter a number between 1 and {len(model_options)}.") is None,
    )
    model_index = (
        int(model_choice_input) - 1
        if model_choice_input
        else model_options.index(_DEFAULT_MODEL_SIZE)
    )
    model_size_str = model_options[model_index]

    # Validate the chosen model size (should always succeed here)
    model_size_validated_result = validate_model_size(model_size_str)
    if model_size_validated_result.is_failure:
        # This indicates an internal logic error
        logging.error(
            f"Internal error validating model size: {model_size_validated_result.error}"
        )
        print("❌ Internal error selecting model size. Exiting.")
        sys.exit(1)
    model_size_validated = model_size_validated_result.unwrap()

    # 6. Confirmation
    print("\n--- Configuration Summary ---")
    print(f"  Input Path:      {input_path_abs}")
    print(f"  Output Path:     {output_path_abs}")
    print(f"  Model Size:      {model_size_validated}")
    if is_input_dir:
        print(f"  Recursive:       {'Yes' if recursive else 'No'}")
        print(f"  Worker Threads:  {worker_count}")
    print("-----------------------------")

    confirm = prompt_user(
        "Proceed with this configuration? (y/n)",
        lambda c: c.lower() in ["y", "yes", "n", "no"]
        or print("❌ Enter 'y' or 'n'.") is None,
    )
    if not confirm.lower().startswith("y"):
        print("🚫 Operation cancelled by user.")
        sys.exit(0)  # Graceful exit on user cancellation

    return (
        input_path_abs,
        output_path_abs,
        model_size_validated,
        recursive,
        worker_count,
    )


def display_statistics() -> None:
    """Retrieves and prints performance statistics from the global registry."""
    stats = metrics_registry.get_all_statistics()
    if not stats:
        print("\n📊 No performance statistics were recorded.")
        return

    print("\n📊 Performance Statistics:")
    # Sort operations alphabetically for consistent output
    for operation, op_stats in sorted(stats.items()):
        if op_stats:
            print(f"\n Operation: {operation}")
            print(f"  ├─ Count:           {op_stats['count']}")
            # Format durations consistently
            print(f"  ├─ Min Duration:    {op_stats['min_duration_ms']:>9.2f} ms")
            print(f"  ├─ Max Duration:    {op_stats['max_duration_ms']:>9.2f} ms")
            print(f"  ├─ Average Duration:{op_stats['avg_duration_ms']:>9.2f} ms")
            print(
                f"  └─ Total Duration:  {op_stats.get('total_duration_ms', 0.0):>9.2f} ms"
            )
        else:
            # Should not happen if get_all_statistics filters empty ops, but handle defensively
            print(f"\n Operation: {operation} (No data recorded)")


# =============================================================================
# 🚀 Main Execution Logic
# =============================================================================


def run_transcription_pipeline(
    input_path: str,
    output_path: str,
    model_size: ModelSize,
    recursive: bool,
    worker_count: int,
) -> Result[Union[str, List[str]]]:
    """
    Orchestrates the transcription process based on input type (file or directory).

    Args:
        input_path: Absolute path to the input file or directory.
        output_path: Absolute path for the output file or directory.
        model_size: Whisper model size (`tiny`, `base`, etc.).
        recursive: Whether to process directories recursively.
        worker_count: Number of workers for parallel directory processing.

    Returns:
        `Result.success(output_path)` for single file success.
        `Result.success(list_of_output_paths)` for directory success (even if partial).
        `Result.failure(Error)` if a fatal error occurs or validation fails early.
        Note: For directory processing, success is returned even if some files fail,
              but the error details are logged. A failure is returned only for
              setup issues or orchestration errors.
    """
    input_p = Path(input_path)

    if input_p.is_dir():
        # --- Directory Processing ---
        output_p = Path(output_path)
        # Check if output path conflicts (e.g., output is an existing file when dir expected)
        if output_p.exists() and not output_p.is_dir():
            return Result.failure(
                Error.create(
                    f"Output path '{output_path}' exists but is not a directory.",
                    "OUTPUT_PATH_NOT_DIR",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"path": output_path},
                )
            )

        dir_params = ProcessDirectoryParams(
            input_dir=input_path,
            output_dir=output_path,
            model_size=model_size,
            recursive=recursive,
            worker_count=worker_count,
        )
        # process_directory returns Result[List[str]]
        dir_result: Result[List[str]] = process_directory(dir_params)
        # Cast to the union type for the return signature
        return cast(Result[Union[str, List[str]]], dir_result)

    elif input_p.is_file():
        # --- Single File Processing ---
        output_p = Path(output_path)
        # Check if output path conflicts (e.g., output is an existing directory when file expected)
        if output_p.exists() and output_p.is_dir():
            return Result.failure(
                Error.create(
                    f"Output path '{output_path}' exists but is a directory. Please specify a file path.",
                    "OUTPUT_PATH_IS_DIR",
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.ERROR,
                    {"path": output_path},
                )
            )
        # Ensure parent directory exists for the output file
        try:
            output_p.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return Result.failure(
                Error.create(
                    f"Cannot create parent directory for output file '{output_path}': {e.strerror}",
                    "OUTPUT_PARENT_DIR_CREATE_FAILED",
                    ErrorCategory.RESOURCE,
                    ErrorSeverity.ERROR,
                    {"path": str(output_p.parent)},
                    exception=e,
                )
            )

        file_params = ProcessFileParams(
            input_path=input_path,
            output_path=output_path,
            model_size=model_size,
        )
        # process_single_file returns Result[str]
        file_result: Result[str] = process_single_file(file_params)
        # Cast to the union type for the return signature
        return cast(Result[Union[str, List[str]]], file_result)
    else:
        # Input path doesn't exist or is not a file/directory
        return Result.failure(
            Error.create(
                f"Input path '{input_path}' is not a valid file or directory.",
                "INVALID_INPUT_PATH",
                ErrorCategory.VALIDATION,
                ErrorSeverity.ERROR,
                {"path": input_path},
            )
        )


def main() -> int:
    """
    Main entry point: parses args, configures logging, runs pipeline, reports results.

    Returns:
        0 on success (all operations completed, though some files in directory
          mode might have failed individually - check logs).
        1 on configuration errors, dependency issues, or fatal pipeline errors.
        130 if interrupted by user (Ctrl+C).
        2 for other unhandled exceptions.
    """
    try:
        args = parse_arguments()
        configure_logging(args.log_level)
        logging.info("Starting Eidosian Transcription Script...")

        # --- Dependency Check ---
        dep_check_result = check_dependencies()
        if dep_check_result.is_failure:
            error = dep_check_result.expect(
                "Dependency check failed unexpectedly"
            )  # Should always have error
            logging.critical(f"Dependency Error: {error.message} (Code: {error.code})")
            if error.trace and logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"Traceback:\n{error.trace}")
            return 1

        # --- Determine Run Mode & Get Parameters ---
        run_interactively = args.input is None  # Interactive if --input is omitted
        final_params: Dict[str, Any]

        if run_interactively:
            try:
                # interactive_mode handles its own SystemExit on cancellation
                i_input, i_output, i_model, i_rec, i_workers = interactive_mode()
                final_params = {
                    "input_path": i_input,
                    "output_path": i_output,
                    "model_size": i_model,
                    "recursive": i_rec,
                    "worker_count": i_workers,
                    "show_stats": True,  # Always show stats in interactive mode
                }
            except SystemExit as e:
                # Catch exit from interactive mode cancellation
                logging.info("Interactive mode cancelled by user.")
                return (
                    int(e.code) if e.code is not None else 0
                )  # Return exit code (0 for clean cancel)
        else:
            # Validate model size from CLI args
            model_val_result = validate_model_size(args.model)
            if model_val_result.is_failure:
                # Error object should exist
                error = model_val_result.error
                logging.error(
                    f"Configuration Error: {error.message if error else 'Unknown model validation error'}"
                )
                return 1

            # Resolve paths to absolute paths
            input_path_abs = str(Path(args.input).resolve())
            # Output path resolution depends on whether input is dir or file
            # This logic is now handled inside run_transcription_pipeline,
            # but we resolve the base path here.
            output_path_abs = str(Path(args.output).resolve())

            final_params = {
                "input_path": input_path_abs,
                "output_path": output_path_abs,  # Pass absolute path
                "model_size": model_val_result.unwrap(),
                "recursive": args.recursive,
                "worker_count": args.workers,
                "show_stats": args.stats,
            }

        # --- Execute Pipeline ---
        logging.info(f"Processing '{final_params['input_path']}'...")
        start_time = time.monotonic()
        pipeline_result = run_transcription_pipeline(
            input_path=final_params["input_path"],
            output_path=final_params["output_path"],
            model_size=final_params["model_size"],
            recursive=final_params["recursive"],
            worker_count=final_params["worker_count"],
        )
        end_time = time.monotonic()
        total_duration = end_time - start_time

        # --- Report Outcome ---
        exit_code = 0
        if pipeline_result.is_success:
            logging.info(
                f"✅ Transcription pipeline completed successfully in {total_duration:.2f} seconds."
            )
            output_val = pipeline_result.unwrap()
            if isinstance(output_val, list):
                logging.info(f"Produced {len(output_val)} transcript file(s).")
                # Optionally list the files if not too many
                # if len(output_val) < 10:
                #     for fpath in output_val: logging.debug(f"  - {fpath}")
            else:  # Single file case
                logging.info(f"Produced transcript: {output_val}")
        else:
            # Pipeline failed (likely setup, validation, or orchestration error)
            error = pipeline_result.error
            if error:  # Check error is not None
                logging.error(
                    f"❌ Transcription pipeline failed after {total_duration:.2f} seconds. "
                    f"Error: [{error.code}] {error.message}"
                )
                if error.trace and logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"Traceback:\n{error.trace}")
            else:
                logging.error(
                    f"❌ Transcription pipeline failed after {total_duration:.2f} seconds with an unknown error."
                )
            exit_code = 1  # Indicate failure

        # --- Display Stats ---
        if final_params["show_stats"]:
            display_statistics()

        logging.info(f"Script finished with exit code {exit_code}.")
        return exit_code

    except KeyboardInterrupt:
        logging.warning("\n🚫 Operation forcefully interrupted by user (Ctrl+C).")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        # Catch-all for truly unexpected errors at the top level
        logging.critical(f"💥 Unhandled top-level exception: {e}", exc_info=True)
        # Optionally create an Error object for logging consistency
        top_level_error = Error.create(
            "Unhandled top-level exception",
            "UNHANDLED_EXCEPTION",
            ErrorCategory.UNEXPECTED,
            ErrorSeverity.FATAL,
            exception=e,
        )
        logging.critical(f"Error Details: {top_level_error}")
        return 2  # General error code


# =============================================================================
# Entry Point Guard
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
