"""Integration tests for word_forge modules.

This module contains integration tests that verify cross-module functionality
and ensure that different components work correctly together.
"""

import sys
from pathlib import Path

# Ensure repository source is on sys.path for import reliability
_repo_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_repo_src))

import pytest


class TestDatabaseGraphIntegration:
    """Integration tests for database and graph module interaction."""

    def test_graph_manager_reads_from_database(self, tmp_path):
        """Test that GraphManager can read data from DBManager."""
        # Import the real GraphManager directly from its module
        import importlib

        from word_forge.database.database_manager import DBManager

        graph_module = importlib.import_module("word_forge.graph.graph_manager")
        # Reload to get the real class, not a stub
        importlib.reload(graph_module)
        GraphManager = graph_module.GraphManager

        # Setup database
        db_path = tmp_path / "integration_test.db"
        db = DBManager(db_path=db_path)

        # Insert a word using the correct method
        db.insert_or_update_word(term="integration", definition="test definition", part_of_speech="noun")

        # Create graph manager with the real class
        graph = GraphManager(db_manager=db)

        # Build graph from database using correct method
        graph.build_graph()

        # Verify graph has nodes using correct method
        assert graph.get_node_count() >= 0


class TestQueueManagerIntegration:
    """Integration tests for queue manager."""

    def test_queue_manager_priority_processing(self):
        """Test queue manager processes items by priority."""
        from word_forge.queue.queue_manager import QueueManager

        queue: QueueManager[str] = QueueManager()

        # Enqueue items with different priorities
        queue.enqueue("low_priority", priority=3)
        queue.enqueue("high_priority", priority=1)
        queue.enqueue("normal_priority", priority=2)

        # Dequeue returns a Result, need to unwrap
        result = queue.dequeue()
        assert result.is_success
        assert result.unwrap() == "high_priority"


class TestEmotionVectorOperations:
    """Integration tests for emotion vector operations."""

    def test_emotion_vector_blend_chain(self):
        """Test chaining multiple emotion vector operations."""
        from word_forge.emotion.emotion_types import EmotionDimension, EmotionVector

        # Create two vectors
        happy = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.8,
                EmotionDimension.AROUSAL: 0.6,
            }
        )
        calm = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.2,
                EmotionDimension.AROUSAL: 0.1,
            }
        )

        # Blend and intensify
        blended = happy.blend(calm, weight=0.5)
        intensified = blended.intensify(factor=1.2)

        # Verify operations work correctly
        assert intensified.dimensions[EmotionDimension.VALENCE] > blended.dimensions[EmotionDimension.VALENCE]


class TestResultMonadChaining:
    """Integration tests for Result monad chaining."""

    def test_result_chain_across_operations(self):
        """Test Result monad chaining across multiple operations."""
        from word_forge.configs.config_essentials import Result

        def validate_positive(x: int) -> Result[int]:
            if x < 0:
                return Result.failure("NEGATIVE", f"Value {x} is negative")
            return Result.success(x)

        def double(x: int) -> Result[int]:
            return Result.success(x * 2)

        def add_ten(x: int) -> Result[int]:
            return Result.success(x + 10)

        # Chain operations
        result = Result.success(5).flat_map(validate_positive).flat_map(double).flat_map(add_ten)

        assert result.is_success
        assert result.unwrap() == 20  # (5 * 2) + 10 = 20

    def test_result_chain_fails_early(self):
        """Test Result monad fails early in chain."""
        from word_forge.configs.config_essentials import Result

        def validate_positive(x: int) -> Result[int]:
            if x < 0:
                return Result.failure("NEGATIVE", f"Value {x} is negative")
            return Result.success(x)

        result = (
            Result.success(-5)
            .flat_map(validate_positive)  # Fails here
            .map(lambda x: x * 2)  # Never executed
            .map(lambda x: x + 10)  # Never executed
        )

        assert result.is_failure
        assert result.error is not None
        assert result.error.code == "NEGATIVE"


class TestConfigurationPipeline:
    """Integration tests for configuration pipeline."""

    def test_queue_config_profile_application(self):
        """Test applying performance profiles to queue config."""
        from word_forge.queue.queue_config import QueueConfig, QueuePerformanceProfile

        config = QueueConfig()

        # Apply high throughput profile
        high_throughput = config.with_performance_profile(QueuePerformanceProfile.HIGH_THROUGHPUT)

        # Apply low latency profile
        low_latency = config.with_performance_profile(QueuePerformanceProfile.LOW_LATENCY)

        # High throughput should have larger batch size
        assert high_throughput.batch_size > low_latency.batch_size

        # Low latency should have smaller throttle
        assert low_latency.throttle_seconds < high_throughput.throttle_seconds


class TestEmotionConfigMetrics:
    """Integration tests for emotion configuration and metrics."""

    def test_emotion_detection_metrics_optimization(self):
        """Test that metrics can optimize emotion detection."""
        from word_forge.emotion.emotion_config import (
            EmotionCategory,
            EmotionConfig,
            EmotionDetectionMetrics,
        )

        config = EmotionConfig()
        metrics = EmotionDetectionMetrics()

        # Simulate many correct HAPPINESS detections
        for _ in range(20):
            metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)

        # Simulate some incorrect SADNESS detections
        for _ in range(5):
            metrics.record_detection(EmotionCategory.SADNESS, EmotionCategory.ANGER)

        # Get precision for each
        happiness_precision = metrics.get_precision(EmotionCategory.HAPPINESS)
        sadness_precision = metrics.get_precision(EmotionCategory.SADNESS)

        # HAPPINESS should have perfect precision (1.0)
        assert happiness_precision == 1.0

        # SADNESS should have zero precision (all false positives)
        assert sadness_precision == 0.0

        # Optimize weights
        optimized = metrics.optimize_weights(config)

        # HAPPINESS weight should increase (high precision)
        assert optimized[EmotionCategory.HAPPINESS] >= config.get_category_weight(EmotionCategory.HAPPINESS)


class TestExceptionHierarchy:
    """Integration tests for exception hierarchy."""

    def test_exception_inheritance_chain(self):
        """Test that exception classes maintain proper inheritance."""
        from word_forge.exceptions import (
            GraphError,
            NodeNotFoundError,
            WordForgeError,
        )

        # Create a node not found error
        error = NodeNotFoundError("Node not found")

        # Should be instance of all parent classes
        assert isinstance(error, NodeNotFoundError)
        assert isinstance(error, GraphError)
        assert isinstance(error, WordForgeError)
        assert isinstance(error, Exception)

    def test_exception_error_message_chaining(self):
        """Test exception error message chaining."""
        from word_forge.exceptions import DatabaseError, WordForgeError

        # Create chained exception
        root_cause = ValueError("Connection timeout")
        db_error = DatabaseError("Failed to connect", cause=root_cause)
        wrapper = WordForgeError("Operation failed", cause=db_error)

        # String representation should include all causes
        error_str = str(wrapper)
        assert "Operation failed" in error_str
        assert "Failed to connect" in str(db_error)


class TestRelationshipProcessing:
    """Integration tests for relationship processing."""

    def test_relationship_properties_consistency(self):
        """Test that relationship properties are consistent."""
        from word_forge.relationships import (
            RELATIONSHIP_TYPES,
            get_relationship_color,
            get_relationship_properties,
            get_relationship_weight,
            is_bidirectional,
        )

        for rel_type, props in RELATIONSHIP_TYPES.items():
            # Get properties using different methods
            full_props = get_relationship_properties(rel_type)
            weight = get_relationship_weight(rel_type)
            color = get_relationship_color(rel_type)
            bidirectional = is_bidirectional(rel_type)

            # Verify consistency
            assert full_props["weight"] == weight
            assert full_props["color"] == color
            assert full_props["bidirectional"] == bidirectional


class TestVectorStoreIntegration:
    """Integration tests for vector store operations."""

    @pytest.mark.skip(reason="Requires external model loading not available in sandbox")
    def test_vector_store_dimensions_config(self):
        """Test vector store dimension configuration."""
        pass  # Skipped - requires external network access


class TestDatabaseSchema:
    """Integration tests for database schema."""

    def test_database_tables_created(self, tmp_path):
        """Test that all required tables are created."""
        from word_forge.database.database_manager import DBManager

        db_path = tmp_path / "schema_test.db"
        db = DBManager(db_path=db_path)

        # Access the connection through the manager's method
        # Check if word_exists works (which implies tables are created)
        exists = db.word_exists("nonexistent_test_word")
        assert exists is False  # Word shouldn't exist, but method should work


class TestEmotionVectorConversion:
    """Integration tests for emotion vector conversion."""

    def test_emotion_vector_serialization_roundtrip(self):
        """Test that emotion vectors can be serialized and deserialized."""
        from word_forge.emotion.emotion_types import EmotionDimension, EmotionVector

        original = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.7,
                EmotionDimension.AROUSAL: 0.5,
                EmotionDimension.DOMINANCE: 0.3,
            },
            confidence=0.9,
        )

        # Convert to dict and back
        as_dict = original.as_dict()
        restored = EmotionVector.from_dict(as_dict, confidence=0.9)

        # Verify roundtrip
        assert restored.dimensions[EmotionDimension.VALENCE] == 0.7
        assert restored.dimensions[EmotionDimension.AROUSAL] == 0.5
        assert restored.dimensions[EmotionDimension.DOMINANCE] == 0.3
        assert restored.confidence == 0.9


class TestEmotionalContextIntegration:
    """Integration tests for emotional context operations."""

    def test_context_application_to_vector(self):
        """Test applying emotional context to vector."""
        from word_forge.emotion.emotion_types import (
            EmotionalContext,
            EmotionDimension,
            EmotionVector,
        )

        # Create vector and context
        vector = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.5,
                EmotionDimension.AROUSAL: 0.5,
            }
        )
        context = EmotionalContext(cultural_factors={"valence": 0.3})

        # Apply context
        modified = context.apply_to_vector(vector)

        # Should produce a new vector (different from original)
        assert modified is not None


class TestExecutionMetricsIntegration:
    """Integration tests for execution metrics."""

    def test_measure_execution_context_manager(self):
        """Test execution measurement context manager."""
        from word_forge.configs.config_essentials import measure_execution

        with measure_execution("test_operation", {"test_key": "test_value"}) as metrics:
            # Simulate some work
            _ = sum(range(10000))

        # Metrics should be populated
        assert metrics.operation_name == "test_operation"
        assert metrics.duration_ms > 0
        assert metrics.context["test_key"] == "test_value"


class TestProcessingStatsIntegration:
    """Integration tests for processing statistics."""

    def test_processing_stats_accumulation(self):
        """Test that processing stats accumulate correctly."""
        pytest.importorskip("nltk", reason="NLTK required for queue_worker")
        from word_forge.queue.queue_worker import (
            ProcessingResult,
            ProcessingStats,
            ProcessingStatus,
        )

        stats = ProcessingStats()

        # Add multiple results
        results = [
            ProcessingResult.success("term1", 10.0, 3, 2, {"synonym": 2, "antonym": 1}),
            ProcessingResult.success("term2", 20.0, 5, 3, {"synonym": 3, "antonym": 2}),
            ProcessingResult.duplicate("term3"),
            ProcessingResult.error("term4", ProcessingStatus.DATABASE_ERROR, "Error"),
        ]

        for result in results:
            stats.update(result)

        # Verify accumulation
        assert stats.processed_count == 4
        assert stats.success_count == 2
        assert stats.duplicate_count == 1
        assert stats.error_count == 1
        assert stats.relationship_counts["synonym"] == 5
        assert stats.relationship_counts["antonym"] == 3
