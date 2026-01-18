"""Tests for word_forge.exceptions module.

This module tests all custom exception classes defined in exceptions.py,
ensuring correct instantiation, inheritance hierarchy, and error message
formatting.
"""

import pytest

from word_forge.exceptions import (
    WordForgeError,
    DatabaseError,
    GraphError,
    GraphAnalysisError,
    GraphExportError,
    GraphImportError,
    GraphUpdateError,
    GraphQueryError,
    GraphConnectionError,
    GraphTraversalError,
    GraphStorageError,
    GraphSerializationError,
    GraphIOError,
    GraphLayoutError,
    QueueError,
    ParserError,
    NodeNotFoundError,
    GraphDataError,
    GraphVisualizationError,
    GraphDimensionError,
)


class TestWordForgeError:
    """Tests for the base WordForgeError class."""

    def test_init_with_message_only(self):
        """Test WordForgeError initialization with just a message."""
        error = WordForgeError("Test error message")
        assert error.message == "Test error message"
        assert error.cause is None
        # Error code is included in string representation
        assert "Test error message" in str(error)
        assert error.error_code in str(error)

    def test_init_with_message_and_cause(self):
        """Test WordForgeError initialization with message and cause."""
        cause = ValueError("Original error")
        error = WordForgeError("Wrapped error", cause=cause)
        assert error.message == "Wrapped error"
        assert error.cause is cause
        assert "Wrapped error" in str(error)
        assert "Original error" in str(error)

    def test_str_representation_without_cause(self):
        """Test __str__ representation without a cause."""
        error = WordForgeError("Simple error")
        # Error code is included in string representation
        assert "Simple error" in str(error)
        assert error.error_code in str(error)

    def test_str_representation_with_cause(self):
        """Test __str__ representation with a cause."""
        cause = RuntimeError("Root cause")
        error = WordForgeError("Wrapper", cause=cause)
        # Error code is included in string representation
        assert "Wrapper" in str(error)
        assert "Root cause" in str(error)
        assert error.error_code in str(error)

    def test_exception_hierarchy(self):
        """Test that WordForgeError inherits from Exception."""
        error = WordForgeError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that WordForgeError can be raised and caught."""
        with pytest.raises(WordForgeError) as exc_info:
            raise WordForgeError("Test error")
        assert "Test error" in str(exc_info.value)


class TestDatabaseError:
    """Tests for DatabaseError class."""

    def test_inherits_from_word_forge_error(self):
        """Test that DatabaseError inherits from WordForgeError."""
        error = DatabaseError("DB error")
        assert isinstance(error, WordForgeError)
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that DatabaseError can be raised and caught."""
        with pytest.raises(DatabaseError):
            raise DatabaseError("Connection failed")


class TestGraphError:
    """Tests for GraphError class."""

    def test_inherits_from_word_forge_error(self):
        """Test that GraphError inherits from WordForgeError."""
        error = GraphError("Graph error")
        assert isinstance(error, WordForgeError)

    def test_can_be_raised_and_caught(self):
        """Test that GraphError can be raised and caught."""
        with pytest.raises(GraphError):
            raise GraphError("Graph operation failed")


class TestGraphErrorSubclasses:
    """Tests for all GraphError subclasses."""

    @pytest.mark.parametrize(
        "error_class",
        [
            GraphAnalysisError,
            GraphExportError,
            GraphImportError,
            GraphUpdateError,
            GraphQueryError,
            GraphConnectionError,
            GraphTraversalError,
            GraphStorageError,
            GraphSerializationError,
            GraphIOError,
            GraphLayoutError,
            NodeNotFoundError,
            GraphDataError,
            GraphVisualizationError,
            GraphDimensionError,
        ],
    )
    def test_inherits_from_graph_error(self, error_class):
        """Test that subclass inherits from GraphError."""
        error = error_class("Test error")
        assert isinstance(error, GraphError)
        assert isinstance(error, WordForgeError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize(
        "error_class",
        [
            GraphAnalysisError,
            GraphExportError,
            GraphImportError,
            GraphUpdateError,
            GraphQueryError,
            GraphConnectionError,
            GraphTraversalError,
            GraphStorageError,
            GraphSerializationError,
            GraphIOError,
            GraphLayoutError,
            NodeNotFoundError,
            GraphDataError,
            GraphVisualizationError,
            GraphDimensionError,
        ],
    )
    def test_can_be_raised_and_caught_as_graph_error(self, error_class):
        """Test that subclass can be caught as GraphError."""
        with pytest.raises(GraphError):
            raise error_class("Test error")


class TestQueueError:
    """Tests for QueueError class."""

    def test_inherits_from_word_forge_error(self):
        """Test that QueueError inherits from WordForgeError."""
        error = QueueError("Queue error")
        assert isinstance(error, WordForgeError)

    def test_can_be_raised_and_caught(self):
        """Test that QueueError can be raised and caught."""
        with pytest.raises(QueueError):
            raise QueueError("Queue operation failed")


class TestParserError:
    """Tests for ParserError class."""

    def test_inherits_from_word_forge_error(self):
        """Test that ParserError inherits from WordForgeError."""
        error = ParserError("Parser error")
        assert isinstance(error, WordForgeError)

    def test_can_be_raised_and_caught(self):
        """Test that ParserError can be raised and caught."""
        with pytest.raises(ParserError):
            raise ParserError("Parse operation failed")


class TestExceptionChaining:
    """Tests for exception chaining functionality."""

    def test_word_forge_error_chains_cause(self):
        """Test that WordForgeError properly chains exceptions."""
        original = ValueError("Original")
        wrapped = WordForgeError("Wrapped", cause=original)
        assert wrapped.__cause__ is original

    def test_nested_exception_chaining(self):
        """Test nested exception chaining."""
        root = ValueError("Root cause")
        middle = DatabaseError("Database issue", cause=root)
        outer = WordForgeError("Application error", cause=middle)

        assert outer.cause is middle
        assert middle.cause is root

    def test_error_message_propagation(self):
        """Test that error messages include cause information."""
        cause = RuntimeError("Network timeout")
        error = DatabaseError("Connection failed", cause=cause)
        error_str = str(error)
        assert "Connection failed" in error_str
        assert "Network timeout" in error_str


class TestSpecificGraphErrors:
    """Tests for specific use cases of graph errors."""

    def test_node_not_found_error(self):
        """Test NodeNotFoundError for missing node scenarios."""
        error = NodeNotFoundError("Node 'test' not found in graph")
        assert "test" in str(error)
        assert isinstance(error, GraphError)

    def test_graph_data_error(self):
        """Test GraphDataError for data inconsistency scenarios."""
        error = GraphDataError("Missing required attribute 'weight' on edge")
        assert "weight" in str(error)
        assert isinstance(error, GraphError)

    def test_graph_visualization_error(self):
        """Test GraphVisualizationError for rendering failures."""
        error = GraphVisualizationError("Failed to generate HTML output")
        assert "HTML" in str(error)
        assert isinstance(error, GraphError)

    def test_graph_dimension_error(self):
        """Test GraphDimensionError for dimension-related failures."""
        error = GraphDimensionError("Cannot project to 4D space")
        assert "4D" in str(error)
        assert isinstance(error, GraphError)
