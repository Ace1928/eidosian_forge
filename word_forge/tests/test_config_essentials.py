"""Tests for word_forge.configs.config_essentials module.

This module tests the core configuration types and utilities including Error,
Result, various enums, and utility functions.
"""

import time
import threading

import pytest

from word_forge.configs.config_essentials import (
    Error,
    ErrorCategory,
    ErrorSeverity,
    Result,
    TaskPriority,
    WorkerState,
    CircuitBreakerState,
    CircuitBreakerConfig,
    ExecutionMetrics,
    TracingContext,
    StorageType,
    measure_execution,
)


class TestErrorCategory:
    """Tests for the ErrorCategory enum."""

    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        assert ErrorCategory.VALIDATION
        assert ErrorCategory.RESOURCE
        assert ErrorCategory.BUSINESS
        assert ErrorCategory.EXTERNAL
        assert ErrorCategory.UNEXPECTED
        assert ErrorCategory.CONFIGURATION
        assert ErrorCategory.SECURITY

    def test_category_count(self):
        """Test the number of categories."""
        assert len(ErrorCategory) == 7


class TestErrorSeverity:
    """Tests for the ErrorSeverity enum."""

    def test_all_severities_exist(self):
        """Test that all expected severities exist."""
        assert ErrorSeverity.FATAL
        assert ErrorSeverity.ERROR
        assert ErrorSeverity.WARNING
        assert ErrorSeverity.INFO

    def test_severity_count(self):
        """Test the number of severity levels."""
        assert len(ErrorSeverity) == 4


class TestError:
    """Tests for the Error dataclass."""

    def test_create_basic_error(self):
        """Test creating a basic error."""
        error = Error(
            message="Test error",
            code="TEST_001",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
        )
        assert error.message == "Test error"
        assert error.code == "TEST_001"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.ERROR
        assert error.context == {}
        assert error.trace is None

    def test_create_error_with_context(self):
        """Test creating an error with context."""
        error = Error(
            message="Error with context",
            code="CTX_001",
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.WARNING,
            context={"key": "value", "another": "data"},
        )
        assert error.context["key"] == "value"
        assert error.context["another"] == "data"

    def test_error_create_factory(self):
        """Test the Error.create factory method."""
        error = Error.create(
            message="Factory error",
            code="FACTORY_001",
            category=ErrorCategory.EXTERNAL,
            severity=ErrorSeverity.FATAL,
            context={"source": "test"},
        )
        assert error.message == "Factory error"
        assert error.code == "FACTORY_001"
        assert error.category == ErrorCategory.EXTERNAL
        assert error.severity == ErrorSeverity.FATAL
        assert error.context["source"] == "test"
        # Factory method should capture traceback
        assert error.trace is not None

    def test_error_is_frozen(self):
        """Test that Error is immutable."""
        error = Error(
            message="Frozen",
            code="FROZEN",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
        )
        with pytest.raises(Exception):
            error.message = "Changed"  # type: ignore


class TestResult:
    """Tests for the Result generic dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = Result.success("value")
        assert result.is_success is True
        assert result.is_failure is False
        assert result.value == "value"
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result: Result[str] = Result.failure(
            code="FAIL_001",
            message="Operation failed",
        )
        assert result.is_success is False
        assert result.is_failure is True
        assert result.value is None
        assert result.error is not None
        assert result.error.code == "FAIL_001"
        assert result.error.message == "Operation failed"

    def test_failure_with_category_and_severity(self):
        """Test creating a failure with custom category and severity."""
        result: Result[int] = Result.failure(
            code="CUSTOM_001",
            message="Custom failure",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.FATAL,
        )
        assert result.error is not None
        assert result.error.category == ErrorCategory.SECURITY
        assert result.error.severity == ErrorSeverity.FATAL

    def test_unwrap_success(self):
        """Test unwrap on successful result."""
        result = Result.success(42)
        assert result.unwrap() == 42

    def test_unwrap_failure_raises(self):
        """Test unwrap on failure raises ValueError."""
        result: Result[int] = Result.failure(
            code="UNWRAP_FAIL",
            message="Cannot unwrap",
        )
        with pytest.raises(ValueError, match="Cannot unwrap failed result"):
            result.unwrap()

    def test_unwrap_or_success(self):
        """Test unwrap_or on successful result."""
        result = Result.success(42)
        assert result.unwrap_or(0) == 42

    def test_unwrap_or_failure(self):
        """Test unwrap_or on failure returns default."""
        result: Result[int] = Result.failure(
            code="DEFAULT",
            message="Use default",
        )
        assert result.unwrap_or(100) == 100

    def test_map_success(self):
        """Test map on successful result."""
        result = Result.success(5)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_success is True
        assert mapped.unwrap() == 10

    def test_map_failure(self):
        """Test map on failure passes through error."""
        result: Result[int] = Result.failure(
            code="MAP_FAIL",
            message="Map failure",
        )
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_failure is True
        assert mapped.error is not None
        assert mapped.error.code == "MAP_FAIL"

    def test_flat_map_success(self):
        """Test flat_map on successful result."""
        result = Result.success(5)
        flat_mapped = result.flat_map(lambda x: Result.success(x * 2))
        assert flat_mapped.is_success is True
        assert flat_mapped.unwrap() == 10

    def test_flat_map_failure_in_input(self):
        """Test flat_map with failure input."""
        result: Result[int] = Result.failure(
            code="FLAT_FAIL",
            message="Flat map failure",
        )
        flat_mapped = result.flat_map(lambda x: Result.success(x * 2))
        assert flat_mapped.is_failure is True

    def test_flat_map_failure_in_function(self):
        """Test flat_map with failure returned by function."""
        result = Result.success(5)
        flat_mapped: Result[int] = result.flat_map(
            lambda x: Result.failure(code="FUNC_FAIL", message="Function failed")
        )
        assert flat_mapped.is_failure is True
        assert flat_mapped.error is not None
        assert flat_mapped.error.code == "FUNC_FAIL"

    def test_result_is_frozen(self):
        """Test that Result is immutable."""
        result = Result.success("test")
        with pytest.raises(Exception):
            result.value = "changed"  # type: ignore


class TestTaskPriority:
    """Tests for the TaskPriority enum."""

    def test_all_priorities_exist(self):
        """Test that all expected priorities exist."""
        assert TaskPriority.CRITICAL
        assert TaskPriority.HIGH
        assert TaskPriority.NORMAL
        assert TaskPriority.LOW
        assert TaskPriority.BACKGROUND

    def test_priority_ordering(self):
        """Test that priorities are ordered correctly."""
        assert TaskPriority.CRITICAL.value < TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value < TaskPriority.NORMAL.value
        assert TaskPriority.NORMAL.value < TaskPriority.LOW.value
        assert TaskPriority.LOW.value < TaskPriority.BACKGROUND.value

    def test_priority_count(self):
        """Test the number of priority levels."""
        assert len(TaskPriority) == 5


class TestWorkerState:
    """Tests for the WorkerState enum."""

    def test_all_states_exist(self):
        """Test that all expected states exist."""
        assert WorkerState.INITIALIZING
        assert WorkerState.IDLE
        assert WorkerState.PROCESSING
        assert WorkerState.PAUSED
        assert WorkerState.STOPPING
        assert WorkerState.STOPPED
        assert WorkerState.ERROR

    def test_state_count(self):
        """Test the number of worker states."""
        assert len(WorkerState) == 7


class TestCircuitBreakerState:
    """Tests for the CircuitBreakerState enum."""

    def test_all_states_exist(self):
        """Test that all expected states exist."""
        assert CircuitBreakerState.CLOSED
        assert CircuitBreakerState.OPEN
        assert CircuitBreakerState.HALF_OPEN

    def test_state_count(self):
        """Test the number of circuit breaker states."""
        assert len(CircuitBreakerState) == 3


class TestCircuitBreakerConfig:
    """Tests for the CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.reset_timeout_ms == 30000
        assert config.half_open_max_calls == 3
        assert config.call_timeout_ms == 5000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            reset_timeout_ms=60000,
            half_open_max_calls=5,
            call_timeout_ms=10000,
        )
        assert config.failure_threshold == 10
        assert config.reset_timeout_ms == 60000
        assert config.half_open_max_calls == 5
        assert config.call_timeout_ms == 10000


class TestExecutionMetrics:
    """Tests for the ExecutionMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = ExecutionMetrics(operation_name="test_op")
        assert metrics.operation_name == "test_op"
        assert metrics.duration_ms == 0.0
        assert metrics.cpu_time_ms == 0.0
        assert metrics.memory_delta_kb == 0.0
        assert metrics.thread_id == 0
        assert metrics.started_at == 0.0
        assert metrics.context == {}

    def test_custom_values(self):
        """Test custom metric values."""
        metrics = ExecutionMetrics(
            operation_name="custom_op",
            duration_ms=100.5,
            cpu_time_ms=50.0,
            memory_delta_kb=1024.0,
            thread_id=12345,
            started_at=time.time(),
            context={"key": "value"},
        )
        assert metrics.operation_name == "custom_op"
        assert metrics.duration_ms == 100.5
        assert metrics.cpu_time_ms == 50.0
        assert metrics.memory_delta_kb == 1024.0
        assert metrics.thread_id == 12345
        assert metrics.context["key"] == "value"


class TestTracingContext:
    """Tests for the TracingContext class."""

    def test_initial_state(self):
        """Test initial state has no context."""
        TracingContext.clear_trace_context()
        assert TracingContext.get_current_trace_id() is None
        assert TracingContext.get_current_span_id() is None

    def test_set_trace_context(self):
        """Test setting trace context."""
        TracingContext.set_trace_context("trace-123", "span-456")
        assert TracingContext.get_current_trace_id() == "trace-123"
        assert TracingContext.get_current_span_id() == "span-456"

    def test_clear_trace_context(self):
        """Test clearing trace context."""
        TracingContext.set_trace_context("trace-123", "span-456")
        TracingContext.clear_trace_context()
        assert TracingContext.get_current_trace_id() is None
        assert TracingContext.get_current_span_id() is None


class TestMeasureExecution:
    """Tests for the measure_execution context manager."""

    def test_basic_measurement(self):
        """Test basic execution measurement."""
        with measure_execution("test_op") as metrics:
            time.sleep(0.01)  # Small delay to measure

        assert metrics.operation_name == "test_op"
        assert metrics.duration_ms > 0
        assert metrics.thread_id == threading.get_ident()
        assert metrics.started_at > 0

    def test_measurement_with_context(self):
        """Test measurement with context."""
        with measure_execution("test_op", {"key": "value"}) as metrics:
            pass

        assert metrics.context["key"] == "value"

    def test_cpu_time_measured(self):
        """Test that CPU time is measured."""
        with measure_execution("cpu_op") as metrics:
            # Do some CPU work
            _ = sum(range(10000))

        assert metrics.cpu_time_ms >= 0


class TestStorageType:
    """Tests for the StorageType enum."""

    def test_storage_types_exist(self):
        """Test that storage types exist."""
        assert StorageType.MEMORY
        assert StorageType.DISK

    def test_storage_type_count(self):
        """Test the number of storage types."""
        assert len(StorageType) == 2


class TestResultChaining:
    """Tests for Result monadic chaining."""

    def test_chain_multiple_maps(self):
        """Test chaining multiple map operations."""
        result = (
            Result.success(5)
            .map(lambda x: x + 1)
            .map(lambda x: x * 2)
            .map(lambda x: x - 2)
        )
        assert result.unwrap() == 10

    def test_chain_stops_on_first_failure(self):
        """Test that chaining stops on first failure."""

        def fail_on_large(x: int) -> Result[int]:
            if x > 10:
                return Result.failure("TOO_LARGE", f"Value {x} too large")
            return Result.success(x)

        result = (
            Result.success(5)
            .map(lambda x: x * 3)  # 15
            .flat_map(fail_on_large)  # Fails here
            .map(lambda x: x * 2)  # Never executed
        )
        assert result.is_failure is True
        assert result.error is not None
        assert result.error.code == "TOO_LARGE"

    def test_complex_chain(self):
        """Test complex chaining with success path."""

        def add_one(x: int) -> Result[int]:
            return Result.success(x + 1)

        def double(x: int) -> Result[int]:
            return Result.success(x * 2)

        result = Result.success(5).flat_map(add_one).flat_map(double).flat_map(add_one)
        assert result.unwrap() == 13  # (5 + 1) * 2 + 1 = 13
