"""Tests for word_forge.queue.queue_worker module.

This module tests the queue worker classes including ProcessingError,
ProcessingResult, ProcessingStats, and related functionality.
"""

import time


from word_forge.queue.queue_worker import (
    ProcessingError,
    ProcessingResult,
    ProcessingStats,
    ProcessingStatus,
)


class TestProcessingError:
    """Tests for the ProcessingError exception class."""

    def test_init_with_message_only(self):
        """Test ProcessingError with just a message."""
        error = ProcessingError("Test error")
        assert error.message == "Test error"
        assert error.cause is None
        assert error.processing_context == {}
        assert error.timestamp > 0

    def test_init_with_cause(self):
        """Test ProcessingError with a cause."""
        cause = ValueError("Original error")
        error = ProcessingError("Wrapped error", cause=cause)
        assert error.message == "Wrapped error"
        assert error.cause is cause

    def test_init_with_context(self):
        """Test ProcessingError with context."""
        context = {"term": "test", "operation": "parse"}
        error = ProcessingError("Error occurred", processing_context=context)
        assert error.processing_context == context

    def test_str_representation_message_only(self):
        """Test __str__ with message only."""
        error = ProcessingError("Simple error")
        assert str(error) == "Simple error"

    def test_str_representation_with_cause(self):
        """Test __str__ with cause."""
        cause = ValueError("Original")
        error = ProcessingError("Wrapper", cause=cause)
        assert "Wrapper" in str(error)
        assert "Original" in str(error)

    def test_str_representation_with_context(self):
        """Test __str__ with context."""
        context = {"term": "test"}
        error = ProcessingError("Error", processing_context=context)
        error_str = str(error)
        assert "Error" in error_str
        assert "term=test" in error_str

    def test_str_representation_with_all(self):
        """Test __str__ with message, cause, and context."""
        cause = RuntimeError("Root")
        context = {"key": "value"}
        error = ProcessingError("Error", cause=cause, processing_context=context)
        error_str = str(error)
        assert "Error" in error_str
        assert "Root" in error_str
        assert "key=value" in error_str

    def test_to_dict(self):
        """Test to_dict conversion."""
        cause = ValueError("Cause")
        context = {"term": "test"}
        error = ProcessingError("Message", cause=cause, processing_context=context)
        result = error.to_dict()
        assert result["message"] == "Message"
        assert result["cause"] == "Cause"
        assert result["context"] == context
        assert isinstance(result["timestamp"], float)

    def test_to_dict_without_cause(self):
        """Test to_dict without cause."""
        error = ProcessingError("Message")
        result = error.to_dict()
        assert result["cause"] is None


class TestProcessingStatus:
    """Tests for the ProcessingStatus enum."""

    def test_success_status(self):
        """Test SUCCESS status exists."""
        assert ProcessingStatus.SUCCESS

    def test_duplicate_status(self):
        """Test DUPLICATE status exists."""
        assert ProcessingStatus.DUPLICATE

    def test_database_error_status(self):
        """Test DATABASE_ERROR status exists."""
        assert ProcessingStatus.DATABASE_ERROR

    def test_parser_error_status(self):
        """Test PARSER_ERROR status exists."""
        assert ProcessingStatus.PARSER_ERROR

    def test_validation_error_status(self):
        """Test VALIDATION_ERROR status exists."""
        assert ProcessingStatus.VALIDATION_ERROR

    def test_general_error_status(self):
        """Test GENERAL_ERROR status exists."""
        assert ProcessingStatus.GENERAL_ERROR

    def test_all_statuses_count(self):
        """Test that there are 6 statuses."""
        assert len(ProcessingStatus) == 6


class TestProcessingResult:
    """Tests for the ProcessingResult dataclass."""

    def test_success_factory(self):
        """Test success factory method."""
        result = ProcessingResult.success(
            term="test",
            duration_ms=100.0,
            relationships_count=5,
            new_terms_count=3,
            relationship_types={"synonym": 3, "antonym": 2},
        )
        assert result.status == ProcessingStatus.SUCCESS
        assert result.term == "test"
        assert result.duration_ms == 100.0
        assert result.relationships_count == 5
        assert result.new_terms_count == 3
        assert result.relationship_types == {"synonym": 3, "antonym": 2}

    def test_duplicate_factory(self):
        """Test duplicate factory method."""
        result = ProcessingResult.duplicate("duplicate_term")
        assert result.status == ProcessingStatus.DUPLICATE
        assert result.term == "duplicate_term"
        assert "already been processed" in result.error_message

    def test_error_factory(self):
        """Test error factory method."""
        result = ProcessingResult.error(
            term="error_term",
            status=ProcessingStatus.DATABASE_ERROR,
            message="Database connection failed",
        )
        assert result.status == ProcessingStatus.DATABASE_ERROR
        assert result.term == "error_term"
        assert result.error_message == "Database connection failed"

    def test_is_success_property_true(self):
        """Test is_success returns True for SUCCESS status."""
        result = ProcessingResult.success(
            term="test",
            duration_ms=100.0,
            relationships_count=0,
            new_terms_count=0,
            relationship_types={},
        )
        assert result.is_success is True

    def test_is_success_property_false(self):
        """Test is_success returns False for other statuses."""
        result = ProcessingResult.duplicate("term")
        assert result.is_success is False

    def test_is_duplicate_property_true(self):
        """Test is_duplicate returns True for DUPLICATE status."""
        result = ProcessingResult.duplicate("term")
        assert result.is_duplicate is True

    def test_is_duplicate_property_false(self):
        """Test is_duplicate returns False for other statuses."""
        result = ProcessingResult.success(
            term="test",
            duration_ms=100.0,
            relationships_count=0,
            new_terms_count=0,
            relationship_types={},
        )
        assert result.is_duplicate is False

    def test_map_on_success(self):
        """Test map transforms term on success."""
        result = ProcessingResult.success(
            term="test",
            duration_ms=100.0,
            relationships_count=1,
            new_terms_count=1,
            relationship_types={"synonym": 1},
        )
        mapped = result.map(lambda t: t.upper())
        assert mapped.term == "TEST"
        assert mapped.is_success is True

    def test_map_preserves_error(self):
        """Test map doesn't transform on error."""
        result = ProcessingResult.duplicate("test")
        mapped = result.map(lambda t: t.upper())
        assert mapped.term == "test"  # Not transformed
        assert mapped.is_duplicate is True

    def test_default_values(self):
        """Test default values for ProcessingResult."""
        result = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            term="test",
        )
        assert result.duration_ms == 0
        assert result.relationships_count == 0
        assert result.new_terms_count == 0
        assert result.error_message is None
        assert result.relationship_types == {}


class TestProcessingStats:
    """Tests for the ProcessingStats dataclass."""

    def test_default_values(self):
        """Test default values."""
        stats = ProcessingStats()
        assert stats.processed_count == 0
        assert stats.success_count == 0
        assert stats.duplicate_count == 0
        assert stats.error_count == 0
        assert stats.total_duration_ms == 0
        assert stats.last_processed is None
        assert stats.relationship_counts == {}

    def test_update_with_success(self):
        """Test update with successful result."""
        stats = ProcessingStats()
        result = ProcessingResult.success(
            term="test",
            duration_ms=50.0,
            relationships_count=3,
            new_terms_count=2,
            relationship_types={"synonym": 2, "antonym": 1},
        )
        stats.update(result)
        assert stats.processed_count == 1
        assert stats.success_count == 1
        assert stats.total_duration_ms == 50.0
        assert stats.last_processed == "test"
        assert stats.relationship_counts == {"synonym": 2, "antonym": 1}

    def test_update_with_duplicate(self):
        """Test update with duplicate result."""
        stats = ProcessingStats()
        result = ProcessingResult.duplicate("duplicate")
        stats.update(result)
        assert stats.processed_count == 1
        assert stats.duplicate_count == 1
        assert stats.success_count == 0
        assert stats.last_processed == "duplicate"

    def test_update_with_error(self):
        """Test update with error result."""
        stats = ProcessingStats()
        result = ProcessingResult.error(
            term="error_term",
            status=ProcessingStatus.DATABASE_ERROR,
            message="Error",
        )
        stats.update(result)
        assert stats.processed_count == 1
        assert stats.error_count == 1
        assert stats.success_count == 0
        assert stats.last_processed == "error_term"

    def test_update_accumulates_relationships(self):
        """Test that relationship counts accumulate."""
        stats = ProcessingStats()
        result1 = ProcessingResult.success(
            term="term1",
            duration_ms=10.0,
            relationships_count=2,
            new_terms_count=1,
            relationship_types={"synonym": 2},
        )
        result2 = ProcessingResult.success(
            term="term2",
            duration_ms=20.0,
            relationships_count=3,
            new_terms_count=2,
            relationship_types={"synonym": 1, "antonym": 2},
        )
        stats.update(result1)
        stats.update(result2)
        assert stats.relationship_counts == {"synonym": 3, "antonym": 2}

    def test_avg_processing_time_ms(self):
        """Test average processing time calculation."""
        stats = ProcessingStats()
        stats.total_duration_ms = 300.0
        stats.processed_count = 3
        assert stats.avg_processing_time_ms == 100.0

    def test_avg_processing_time_ms_zero(self):
        """Test average processing time with no items."""
        stats = ProcessingStats()
        assert stats.avg_processing_time_ms == 0.0

    def test_processing_rate_per_minute(self):
        """Test processing rate calculation."""
        stats = ProcessingStats()
        stats.processed_count = 60
        # Set start_time to 1 second ago
        stats.start_time = time.time() - 1
        # Should be approximately 3600 per minute (60 / 1 sec * 60)
        rate = stats.processing_rate_per_minute
        assert 3500 < rate < 3700

    def test_processing_rate_per_minute_zero_time(self):
        """Test processing rate with very short time."""
        stats = ProcessingStats()
        stats.processed_count = 10
        stats.start_time = time.time()  # Just now
        # Should return 0 for very short time
        assert stats.processing_rate_per_minute >= 0

    def test_reset(self):
        """Test reset clears all statistics."""
        stats = ProcessingStats()
        stats.processed_count = 10
        stats.success_count = 8
        stats.duplicate_count = 1
        stats.error_count = 1
        stats.total_duration_ms = 500.0
        stats.last_processed = "last"
        stats.relationship_counts = {"synonym": 5}

        stats.reset()

        assert stats.processed_count == 0
        assert stats.success_count == 0
        assert stats.duplicate_count == 0
        assert stats.error_count == 0
        assert stats.total_duration_ms == 0
        assert stats.last_processed is None
        assert stats.relationship_counts == {}


class TestProcessingResultMapChaining:
    """Tests for Result monad-style chaining."""

    def test_chain_multiple_maps(self):
        """Test chaining multiple map operations."""
        result = ProcessingResult.success(
            term="test",
            duration_ms=10.0,
            relationships_count=1,
            new_terms_count=0,
            relationship_types={},
        )
        final = result.map(str.upper).map(lambda s: s + "_suffix")
        assert final.term == "TEST_suffix"

    def test_chain_stops_on_error(self):
        """Test that chaining stops on error result."""
        result = ProcessingResult.error(
            term="error",
            status=ProcessingStatus.VALIDATION_ERROR,
            message="Invalid",
        )
        final = result.map(str.upper).map(lambda s: s + "_suffix")
        assert final.term == "error"  # Not transformed
        assert final.status == ProcessingStatus.VALIDATION_ERROR
