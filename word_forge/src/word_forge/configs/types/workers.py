"""
Worker and task management types for Word Forge configuration system.

This module provides types for task scheduling, worker state management,
circuit breaker patterns, and execution metrics.

Classes:
    TaskPriority: Priority levels for task scheduling
    WorkerState: States for worker threads
    CircuitBreakerState: States for circuit breaker pattern
    ExecutionMetrics: Metrics for operation execution
    CircuitBreakerConfig: Configuration for circuit breakers
    TracingContext: Thread-local distributed tracing

Functions:
    measure_execution: Context manager for execution metrics
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
)

try:
    from eidosian_core import eidosian
except ImportError:
    def eidosian():
        """Fallback decorator when eidosian_core is not available."""
        def decorator(func):
            return func
        return decorator


class TaskPriority(Enum):
    """Priority levels for scheduling tasks in the work distribution system."""

    CRITICAL = 0  # Must be processed immediately
    HIGH = 1  # Process before normal tasks
    NORMAL = 2  # Default priority
    LOW = 3  # Process after other tasks
    BACKGROUND = 4  # Process only when system is idle


class WorkerState(Enum):
    """States for worker threads in the processing system."""

    INITIALIZING = auto()  # Worker is initializing resources
    IDLE = auto()  # Worker is waiting for tasks
    PROCESSING = auto()  # Worker is processing a task
    PAUSED = auto()  # Worker is temporarily paused
    STOPPING = auto()  # Worker is in the process of stopping
    STOPPED = auto()  # Worker has stopped
    ERROR = auto()  # Worker encountered an error


class CircuitBreakerState(Enum):
    """States for the circuit breaker pattern to prevent cascading failures."""

    CLOSED = auto()  # Normal operation, requests allowed
    OPEN = auto()  # Failing, rejecting requests
    HALF_OPEN = auto()  # Testing if system has recovered


@dataclass
class ExecutionMetrics:
    """
    Metrics collected during operation execution for performance monitoring.

    Provides detailed information about execution time and resource usage
    to help identify bottlenecks and performance issues.

    Attributes:
        operation_name: Name of the operation being measured
        duration_ms: Wall clock execution time in milliseconds
        cpu_time_ms: CPU time used in milliseconds
        memory_delta_kb: Memory usage change in kilobytes
        thread_id: ID of the thread executing the operation
        started_at: Timestamp when execution started
        context: Dictionary of additional context for the operation
    """

    operation_name: str
    duration_ms: float = 0.0
    cpu_time_ms: float = 0.0
    memory_delta_kb: float = 0.0
    thread_id: int = 0
    started_at: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for a circuit breaker to prevent system overload.

    Controls the behavior of the circuit breaker pattern implementation
    that protects system components from cascading failures.

    Attributes:
        failure_threshold: Number of failures before circuit opens
        reset_timeout_ms: How long to wait before testing circuit again
        half_open_max_calls: Maximum calls allowed in half-open state
        call_timeout_ms: Timeout for individual calls
    """

    failure_threshold: int = 5
    reset_timeout_ms: int = 30000
    half_open_max_calls: int = 3
    call_timeout_ms: int = 5000


class TracingContext:
    """
    Thread-local storage for distributed tracing information.

    Maintains trace and span IDs across async boundaries to enable
    distributed request tracing throughout the system.
    """

    _thread_local = threading.local()

    @classmethod
    def get_current_trace_id(cls) -> Optional[str]:
        """Get the current trace ID from thread-local storage."""
        return getattr(cls._thread_local, "trace_id", None)

    @classmethod
    def get_current_span_id(cls) -> Optional[str]:
        """Get the current span ID from thread-local storage."""
        return getattr(cls._thread_local, "span_id", None)

    @classmethod
    def set_trace_context(cls, trace_id: str, span_id: str) -> None:
        """
        Set the current trace and span IDs.

        Args:
            trace_id: Trace identifier for the current request
            span_id: Span identifier for the current operation
        """
        cls._thread_local.trace_id = trace_id
        cls._thread_local.span_id = span_id

    @classmethod
    def clear_trace_context(cls) -> None:
        """Clear the current trace context."""
        cls._thread_local.trace_id = None
        cls._thread_local.span_id = None


@eidosian()
@contextmanager
def measure_execution(
    operation_name: str, context: Optional[Dict[str, Any]] = None
) -> Iterator[ExecutionMetrics]:
    """
    Context manager to measure execution time and resource usage.

    Tracks wall clock time, CPU time, and optionally memory usage for
    performance monitoring of critical operations.

    Args:
        operation_name: Unique identifier for the operation
        context: Optional contextual information

    Yields:
        ExecutionMetrics object that will be populated when context exits
    """
    metrics = ExecutionMetrics(
        operation_name=operation_name,
        thread_id=threading.get_ident(),
        started_at=time.time(),
        context=context or {},
    )

    start_time = time.time()
    start_cpu_time = time.process_time()

    try:
        yield metrics
    finally:
        metrics.duration_ms = (time.time() - start_time) * 1000
        metrics.cpu_time_ms = (time.process_time() - start_cpu_time) * 1000


__all__ = [
    "TaskPriority",
    "WorkerState",
    "CircuitBreakerState",
    "ExecutionMetrics",
    "CircuitBreakerConfig",
    "TracingContext",
    "measure_execution",
]
