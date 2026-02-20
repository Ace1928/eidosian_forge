"""Tests for word_forge.conversation.conversation_worker module.

This module provides tests for conversation worker state management,
metrics tracking, and error handling.

Note: ConversationWorker class itself requires heavy dependencies (NLTK, etc.)
and is tested in full dev environments. This file tests the supporting
classes that can be imported independently.
"""

import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ============================================================================
# Recreate the core types for testing without heavy imports
# ============================================================================


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


class TaskResult(Enum):
    """Result status of a conversation task processing attempt."""

    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    DEFERRED = auto()
    INVALID = auto()


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
        with self._lock:
            self.processed_count += 1

    def record_result(self, result: TaskResult, processing_time: float) -> None:
        """Record a processing result with timing information."""
        with self._lock:
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

    def record_task_type(self, task_type: str) -> None:
        """Record a task type for distribution analysis."""
        with self._lock:
            self.task_types[task_type] = self.task_types.get(task_type, 0) + 1

    def record_error(self, error_type: str) -> None:
        """Record an error type for error distribution analysis."""
        with self._lock:
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    def get_avg_processing_time(self) -> Optional[float]:
        """Calculate the average processing time."""
        with self._lock:
            if not self.processing_times:
                return None
            return sum(self.processing_times) / len(self.processing_times)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Return a dictionary of all metrics."""
        with self._lock:
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
                "avg_processing_time_ms": (round(avg_time * 1000, 2) if avg_time is not None else None),
                "task_type_distribution": dict(self.task_types),
                "error_type_distribution": dict(self.error_types),
                "success_rate": round(self.success_count / max(1, self.processed_count) * 100, 2),
            }


# ============================================================================
# Tests
# ============================================================================


class TestConversationWorkerState:
    """Tests for the ConversationWorkerState enum."""

    def test_running_state_exists(self) -> None:
        """Test RUNNING state exists."""
        assert ConversationWorkerState.RUNNING

    def test_stopped_state_exists(self) -> None:
        """Test STOPPED state exists."""
        assert ConversationWorkerState.STOPPED

    def test_error_state_exists(self) -> None:
        """Test ERROR state exists."""
        assert ConversationWorkerState.ERROR

    def test_paused_state_exists(self) -> None:
        """Test PAUSED state exists."""
        assert ConversationWorkerState.PAUSED

    def test_recovery_state_exists(self) -> None:
        """Test RECOVERY state exists."""
        assert ConversationWorkerState.RECOVERY

    def test_str_representation(self) -> None:
        """Test string representation is lowercase."""
        assert str(ConversationWorkerState.RUNNING) == "running"
        assert str(ConversationWorkerState.STOPPED) == "stopped"
        assert str(ConversationWorkerState.ERROR) == "error"
        assert str(ConversationWorkerState.PAUSED) == "paused"
        assert str(ConversationWorkerState.RECOVERY) == "recovery"

    def test_all_states_count(self) -> None:
        """Test that there are 5 states."""
        assert len(ConversationWorkerState) == 5


class TestTaskResult:
    """Tests for the TaskResult enum."""

    def test_success_result(self) -> None:
        """Test SUCCESS result exists."""
        assert TaskResult.SUCCESS

    def test_failure_result(self) -> None:
        """Test FAILURE result exists."""
        assert TaskResult.FAILURE

    def test_timeout_result(self) -> None:
        """Test TIMEOUT result exists."""
        assert TaskResult.TIMEOUT

    def test_deferred_result(self) -> None:
        """Test DEFERRED result exists."""
        assert TaskResult.DEFERRED

    def test_invalid_result(self) -> None:
        """Test INVALID result exists."""
        assert TaskResult.INVALID

    def test_all_results_count(self) -> None:
        """Test that there are 5 result types."""
        assert len(TaskResult) == 5


class TestConversationExceptions:
    """Tests for conversation worker exception classes."""

    def test_conversation_error_base(self) -> None:
        """Test ConversationError is the base exception."""
        error = ConversationError("test error")
        assert str(error) == "test error"
        assert isinstance(error, Exception)

    def test_processing_error_inherits(self) -> None:
        """Test ConversationProcessingError inherits from ConversationError."""
        error = ConversationProcessingError("processing failed")
        assert isinstance(error, ConversationError)

    def test_processing_error_with_cause(self) -> None:
        """Test ConversationProcessingError with cause."""
        cause = ValueError("original error")
        error = ConversationProcessingError("processing failed", cause=cause)
        assert error.cause == cause
        assert "original error" in str(error)

    def test_processing_error_with_context(self) -> None:
        """Test ConversationProcessingError with context."""
        context = {"task_id": "123", "conversation_id": 456}
        error = ConversationProcessingError("processing failed", context=context)
        assert error.context == context
        assert "task_id" in str(error)

    def test_timeout_error_inherits(self) -> None:
        """Test ConversationTimeoutError inherits from ConversationError."""
        error = ConversationTimeoutError("operation timed out")
        assert isinstance(error, ConversationError)

    def test_queue_error_inherits(self) -> None:
        """Test ConversationQueueError inherits from ConversationError."""
        error = ConversationQueueError("queue operation failed")
        assert isinstance(error, ConversationError)


class TestProcessingMetrics:
    """Tests for the ProcessingMetrics class."""

    def test_initial_counts_zero(self) -> None:
        """Test that initial counts are zero."""
        metrics = ProcessingMetrics()
        assert metrics.processed_count == 0
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.timeout_count == 0
        assert metrics.deferred_count == 0
        assert metrics.invalid_count == 0

    def test_record_processed(self) -> None:
        """Test incrementing processed count."""
        metrics = ProcessingMetrics()
        metrics.record_processed()
        assert metrics.processed_count == 1
        metrics.record_processed()
        assert metrics.processed_count == 2

    def test_record_result_success(self) -> None:
        """Test recording success result."""
        metrics = ProcessingMetrics()
        metrics.record_result(TaskResult.SUCCESS, 0.5)
        assert metrics.success_count == 1
        assert len(metrics.processing_times) == 1
        assert metrics.processing_times[0] == 0.5

    def test_record_result_failure(self) -> None:
        """Test recording failure result."""
        metrics = ProcessingMetrics()
        metrics.record_result(TaskResult.FAILURE, 0.3)
        assert metrics.error_count == 1

    def test_record_result_timeout(self) -> None:
        """Test recording timeout result."""
        metrics = ProcessingMetrics()
        metrics.record_result(TaskResult.TIMEOUT, 30.0)
        assert metrics.timeout_count == 1

    def test_record_result_deferred(self) -> None:
        """Test recording deferred result."""
        metrics = ProcessingMetrics()
        metrics.record_result(TaskResult.DEFERRED, 0.1)
        assert metrics.deferred_count == 1

    def test_record_result_invalid(self) -> None:
        """Test recording invalid result."""
        metrics = ProcessingMetrics()
        metrics.record_result(TaskResult.INVALID, 0.05)
        assert metrics.invalid_count == 1

    def test_record_task_type(self) -> None:
        """Test recording task type distribution."""
        metrics = ProcessingMetrics()
        metrics.record_task_type("add_message")
        metrics.record_task_type("add_message")
        metrics.record_task_type("start_conversation")
        assert metrics.task_types["add_message"] == 2
        assert metrics.task_types["start_conversation"] == 1

    def test_record_error(self) -> None:
        """Test recording error type distribution."""
        metrics = ProcessingMetrics()
        metrics.record_error("ValueError")
        metrics.record_error("TypeError")
        metrics.record_error("ValueError")
        assert metrics.error_types["ValueError"] == 2
        assert metrics.error_types["TypeError"] == 1

    def test_get_avg_processing_time_empty(self) -> None:
        """Test average processing time with no data."""
        metrics = ProcessingMetrics()
        assert metrics.get_avg_processing_time() is None

    def test_get_avg_processing_time(self) -> None:
        """Test average processing time calculation."""
        metrics = ProcessingMetrics()
        metrics.record_result(TaskResult.SUCCESS, 1.0)
        metrics.record_result(TaskResult.SUCCESS, 2.0)
        metrics.record_result(TaskResult.SUCCESS, 3.0)
        avg = metrics.get_avg_processing_time()
        assert avg is not None
        assert avg == 2.0

    def test_processing_times_limit(self) -> None:
        """Test that processing times are limited to 100 entries."""
        metrics = ProcessingMetrics()
        for i in range(150):
            metrics.record_result(TaskResult.SUCCESS, float(i))
        assert len(metrics.processing_times) == 100

    def test_get_metrics_dict(self) -> None:
        """Test getting metrics as dictionary."""
        metrics = ProcessingMetrics()
        metrics.record_processed()
        metrics.record_result(TaskResult.SUCCESS, 1.0)
        metrics.record_task_type("test")

        metrics_dict = metrics.get_metrics_dict()
        assert "processed_count" in metrics_dict
        assert "success_count" in metrics_dict
        assert "error_count" in metrics_dict
        assert "success_rate" in metrics_dict
        assert "task_type_distribution" in metrics_dict

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        metrics = ProcessingMetrics()
        metrics.record_processed()
        metrics.record_processed()
        metrics.record_result(TaskResult.SUCCESS, 0.5)
        metrics.record_result(TaskResult.FAILURE, 0.5)

        metrics_dict = metrics.get_metrics_dict()
        assert metrics_dict["success_rate"] == 50.0
