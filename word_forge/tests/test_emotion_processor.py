"""Tests for word_forge.emotion.emotion_processor module.

This module provides comprehensive tests for the RecursiveEmotionProcessor class
including emotion extraction, relationship analysis, context management,
and meta-emotion generation.
"""

import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.emotion.emotion_processor import RecursiveEmotionProcessor
from word_forge.emotion.emotion_types import (
    EmotionalConcept,
    EmotionalContext,
    EmotionDimension,
    EmotionVector,
)


def _create_dependencies(tmp_path: Path) -> Tuple[DBManager, EmotionManager]:
    db_manager = DBManager(db_path=tmp_path / "test.db")
    db_manager.create_tables()
    emotion_manager = EmotionManager(db_manager)
    return db_manager, emotion_manager


def _create_processor(
    db_manager: DBManager, emotion_manager: EmotionManager
) -> RecursiveEmotionProcessor:
    return RecursiveEmotionProcessor(db_manager, emotion_manager)


class TestRecursiveEmotionProcessorInit:
    """Tests for RecursiveEmotionProcessor initialization."""

    def test_init_with_managers(self, tmp_path: Path) -> None:
        """Test initialization with DBManager and EmotionManager."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        assert processor.db is db_manager
        assert processor.emotion_manager is emotion_manager
        assert processor._processing_depth == 0
        assert processor._max_recursion == 3
        assert processor._cache == {}
        assert processor.context_registry == {}

    def test_init_creates_hook_registries(self, tmp_path: Path) -> None:
        """Test that initialization creates hook registries."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        assert hasattr(processor, "meta_emotion_hooks")
        assert hasattr(processor, "pattern_hooks")
        assert isinstance(processor.meta_emotion_hooks, list)
        assert isinstance(processor.pattern_hooks, list)


class TestRecursiveScope:
    """Tests for the recursive scope context manager."""

    def test_recursive_scope_increments_depth(self, tmp_path: Path) -> None:
        """Test that recursive scope increments processing depth."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        assert processor._processing_depth == 0
        with processor._recursive_scope():
            assert processor._processing_depth == 1
            with processor._recursive_scope():
                assert processor._processing_depth == 2
            assert processor._processing_depth == 1
        assert processor._processing_depth == 0

    def test_recursive_scope_decrements_on_exception(self, tmp_path: Path) -> None:
        """Test that recursive scope decrements depth even on exception."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        try:
            with processor._recursive_scope():
                assert processor._processing_depth == 1
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert processor._processing_depth == 0


class TestContextManagement:
    """Tests for emotional context management."""

    def test_register_context(self, tmp_path: Path) -> None:
        """Test registering a named context."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = EmotionalContext()
        processor.register_context("test_context", context)
        assert "test_context" in processor.context_registry
        assert processor.context_registry["test_context"] is context

    def test_get_context_exists(self, tmp_path: Path) -> None:
        """Test retrieving an existing context."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = EmotionalContext()
        processor.register_context("existing", context)
        retrieved = processor.get_context("existing")
        assert retrieved is context

    def test_get_context_not_found(self, tmp_path: Path) -> None:
        """Test retrieving a non-existent context returns None."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        assert processor.get_context("nonexistent") is None

    def test_create_context_for_academic_domain(self, tmp_path: Path) -> None:
        """Test creating context for academic domain."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = processor.create_context_for_domain("academic")
        assert context.domain_specific is not None
        assert "valence" in context.domain_specific
        assert context.domain_specific["certainty"] == 0.7

    def test_create_context_for_casual_domain(self, tmp_path: Path) -> None:
        """Test creating context for casual domain."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = processor.create_context_for_domain("casual")
        assert context.domain_specific is not None
        assert context.domain_specific["social"] == 0.5

    def test_create_context_for_medical_domain(self, tmp_path: Path) -> None:
        """Test creating context for medical domain."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = processor.create_context_for_domain("medical")
        assert context.domain_specific is not None
        assert context.domain_specific["certainty"] == 0.8

    def test_create_context_for_literary_domain(self, tmp_path: Path) -> None:
        """Test creating context for literary domain."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = processor.create_context_for_domain("literary")
        assert context.domain_specific is not None
        assert context.domain_specific["novelty"] == 0.7

    def test_create_context_for_unknown_domain(self, tmp_path: Path) -> None:
        """Test creating context for unknown domain returns empty context."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = processor.create_context_for_domain("unknown_domain")
        assert isinstance(context, EmotionalContext)


class TestHeuristicEmotionGeneration:
    """Tests for heuristic emotion generation."""

    def test_generate_heuristic_positive_term(self, tmp_path: Path) -> None:
        """Test heuristic generation for positive terms."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion = processor._generate_heuristic_emotion("happiness")
        assert isinstance(emotion, EmotionVector)
        assert EmotionDimension.VALENCE in emotion.dimensions
        assert emotion.dimensions[EmotionDimension.VALENCE] > 0

    def test_generate_heuristic_negative_term(self, tmp_path: Path) -> None:
        """Test heuristic generation for negative terms."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion = processor._generate_heuristic_emotion("sadness")
        assert isinstance(emotion, EmotionVector)
        assert EmotionDimension.VALENCE in emotion.dimensions
        assert emotion.dimensions[EmotionDimension.VALENCE] < 0

    def test_generate_heuristic_high_arousal_term(self, tmp_path: Path) -> None:
        """Test heuristic generation for high arousal terms."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion = processor._generate_heuristic_emotion("excitement")
        assert isinstance(emotion, EmotionVector)
        assert EmotionDimension.AROUSAL in emotion.dimensions

    def test_generate_heuristic_high_dominance_term(self, tmp_path: Path) -> None:
        """Test heuristic generation for high dominance terms."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion = processor._generate_heuristic_emotion("powerful")
        assert isinstance(emotion, EmotionVector)
        assert EmotionDimension.DOMINANCE in emotion.dimensions
        assert emotion.dimensions[EmotionDimension.DOMINANCE] > 0

    def test_generate_heuristic_returns_valid_confidence(self, tmp_path: Path) -> None:
        """Test that heuristic generation returns valid confidence."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion = processor._generate_heuristic_emotion("test")
        assert 0.0 <= emotion.confidence <= 1.0


class TestDefaultEmotion:
    """Tests for default emotion creation."""

    def test_create_default_emotion_empty_term(self, tmp_path: Path) -> None:
        """Test creating default emotion for empty term."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion = processor._create_default_emotion("")
        assert isinstance(emotion, EmotionVector)
        assert emotion.confidence == 0.5
        assert emotion.dimensions[EmotionDimension.VALENCE] == 0.0
        assert emotion.dimensions[EmotionDimension.AROUSAL] == 0.0
        assert emotion.dimensions[EmotionDimension.DOMINANCE] == 0.0

    def test_create_default_emotion_with_term(self, tmp_path: Path) -> None:
        """Test creating default emotion for a term uses heuristics."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion = processor._create_default_emotion("test_word")
        assert isinstance(emotion, EmotionVector)
        assert emotion.confidence > 0


class TestFallbackConcept:
    """Tests for fallback concept creation."""

    def test_create_fallback_concept(self, tmp_path: Path) -> None:
        """Test creating fallback concept."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        concept = processor._create_fallback_concept("fallback_term")
        assert isinstance(concept, EmotionalConcept)
        assert concept.term == "fallback_term"
        assert concept.word_id == -1
        assert isinstance(concept.primary_emotion, EmotionVector)


class TestNormalizeDimensions:
    """Tests for dimension normalization."""

    def test_normalize_dimensions_within_range(self, tmp_path: Path) -> None:
        """Test normalizing dimensions already within range."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        dimensions = {
            EmotionDimension.VALENCE: 0.5,
            EmotionDimension.AROUSAL: -0.3,
        }
        normalized = processor._normalize_dimensions(dimensions)
        assert normalized[EmotionDimension.VALENCE] == 0.5
        assert normalized[EmotionDimension.AROUSAL] == -0.3

    def test_normalize_dimensions_clips_high(self, tmp_path: Path) -> None:
        """Test normalizing clips values above 1.0."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        dimensions = {EmotionDimension.VALENCE: 2.5}
        normalized = processor._normalize_dimensions(dimensions)
        assert normalized[EmotionDimension.VALENCE] == 1.0

    def test_normalize_dimensions_clips_low(self, tmp_path: Path) -> None:
        """Test normalizing clips values below -1.0."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        dimensions = {EmotionDimension.VALENCE: -2.5}
        normalized = processor._normalize_dimensions(dimensions)
        assert normalized[EmotionDimension.VALENCE] == -1.0


class TestCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_without_context(self, tmp_path: Path) -> None:
        """Test cache key without context is just the term."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        key = processor._get_cache_key("test_term")
        assert key == "test_term"

    def test_cache_key_with_context(self, tmp_path: Path) -> None:
        """Test cache key with context includes hash."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = EmotionalContext()
        context.domain_specific = {"valence": 0.5}
        key = processor._get_cache_key("test_term", context)
        assert key.startswith("test_term::")
        assert len(key) > len("test_term::")

    def test_cache_key_same_for_same_context(self, tmp_path: Path) -> None:
        """Test cache key is deterministic for same context."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        context = EmotionalContext()
        context.domain_specific = {"valence": 0.5}
        key1 = processor._get_cache_key("test", context)
        key2 = processor._get_cache_key("test", context)
        assert key1 == key2


class TestHookRegistration:
    """Tests for hook registration."""

    def test_register_meta_emotion_hook(self, tmp_path: Path) -> None:
        """Test registering a meta-emotion hook."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        initial_count = len(processor.meta_emotion_hooks)

        def custom_hook(concept: EmotionalConcept) -> None:
            pass

        processor.register_meta_emotion_hook(custom_hook)
        assert len(processor.meta_emotion_hooks) == initial_count + 1
        assert custom_hook in processor.meta_emotion_hooks

    def test_register_pattern_hook(self, tmp_path: Path) -> None:
        """Test registering a pattern hook."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        initial_count = len(processor.pattern_hooks)

        def custom_pattern_hook(concept: EmotionalConcept) -> None:
            pass

        processor.register_pattern_hook(custom_pattern_hook)
        assert len(processor.pattern_hooks) == initial_count + 1
        assert custom_pattern_hook in processor.pattern_hooks


class TestRelationshipAnalysis:
    """Tests for relationship analysis methods."""

    def test_calculate_evocative_strength(self, tmp_path: Path) -> None:
        """Test evocative strength calculation."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        emotion1 = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: -0.5,
                EmotionDimension.AROUSAL: 0.3,
            }
        )
        emotion2 = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.2,
                EmotionDimension.AROUSAL: 0.7,
            }
        )

        strength = processor._calculate_evocative_strength(emotion1, emotion2)
        assert 0.0 <= strength <= 1.0

    def test_calculate_component_strength(self, tmp_path: Path) -> None:
        """Test component strength calculation."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        component = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.8,
            }
        )
        composite = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.7,
                EmotionDimension.AROUSAL: 0.5,
                EmotionDimension.DOMINANCE: 0.3,
            }
        )

        strength = processor._calculate_component_strength(component, composite)
        assert 0.0 <= strength <= 1.0


class TestAnalyzeRelationship:
    """Tests for analyze_relationship method."""

    def test_analyze_synonym_relationship(self, tmp_path: Path) -> None:
        """Test analyzing synonym relationships."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word("happy", "feeling happy", "adjective")
        db_manager.insert_or_update_word("joyful", "feeling joyful", "adjective")
        processor = _create_processor(db_manager, emotion_manager)

        strength = processor.analyze_relationship("happy", "joyful", "synonym")
        assert 0.0 <= strength <= 1.0

    def test_analyze_antonym_relationship(self, tmp_path: Path) -> None:
        """Test analyzing antonym relationships."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word("happy", "feeling happy", "adjective")
        db_manager.insert_or_update_word("sad", "feeling sad", "adjective")
        processor = _create_processor(db_manager, emotion_manager)

        strength = processor.analyze_relationship("happy", "sad", "antonym")
        assert 0.0 <= strength <= 1.0

    def test_analyze_intensifies_relationship(self, tmp_path: Path) -> None:
        """Test analyzing intensifies relationships."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word("angry", "feeling angry", "adjective")
        db_manager.insert_or_update_word("furious", "extremely angry", "adjective")
        processor = _create_processor(db_manager, emotion_manager)

        strength = processor.analyze_relationship("angry", "furious", "intensifies")
        assert 0.0 <= strength <= 1.0


class TestGetEmotionVector:
    """Tests for get_emotion_vector method."""

    def test_get_emotion_vector_unknown_term(self, tmp_path: Path) -> None:
        """Test getting emotion vector for unknown term."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        vector = processor.get_emotion_vector("unknown_term_xyz")
        assert isinstance(vector, EmotionVector)

    def test_get_emotion_vector_known_term(self, tmp_path: Path) -> None:
        """Test getting emotion vector for known term."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word("happiness", "state of being happy", "noun")
        word_id = db_manager.get_word_id("happiness")
        emotion_manager.set_word_emotion(word_id, 0.8, 0.6)
        processor = _create_processor(db_manager, emotion_manager)

        vector = processor.get_emotion_vector("happiness")
        assert isinstance(vector, EmotionVector)
        assert EmotionDimension.VALENCE in vector.dimensions
        assert vector.dimensions[EmotionDimension.VALENCE] == 0.8


class TestProcessTerm:
    """Tests for process_term method."""

    def test_process_term_returns_concept(self, tmp_path: Path) -> None:
        """Test process_term returns EmotionalConcept."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word("love", "deep affection", "noun")
        processor = _create_processor(db_manager, emotion_manager)

        concept = processor.process_term("love")
        assert isinstance(concept, EmotionalConcept)
        assert concept.term == "love"
        assert isinstance(concept.primary_emotion, EmotionVector)

    def test_process_term_caches_result(self, tmp_path: Path) -> None:
        """Test process_term caches the result."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word("test", "test word", "noun")
        processor = _create_processor(db_manager, emotion_manager)

        concept1 = processor.process_term("test")
        assert "test" in processor._cache
        concept2 = processor.process_term("test")
        assert concept1 is concept2

    def test_process_term_respects_recursion_limit(self, tmp_path: Path) -> None:
        """Test process_term respects recursion limit."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        processor = _create_processor(db_manager, emotion_manager)

        processor._processing_depth = processor._max_recursion + 1
        concept = processor.process_term("deep_term")
        assert concept.word_id == -1


class TestIntegration:
    """Integration tests for RecursiveEmotionProcessor."""

    def test_full_processing_pipeline(self, tmp_path: Path) -> None:
        """Test full processing pipeline."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word(
            "euphoria", "intense feeling of happiness", "noun"
        )
        word_id = db_manager.get_word_id("euphoria")
        emotion_manager.set_word_emotion(word_id, 0.9, 0.8)
        processor = _create_processor(db_manager, emotion_manager)

        concept = processor.process_term("euphoria")

        assert concept.term == "euphoria"
        assert concept.word_id == word_id
        assert concept.primary_emotion.dimensions[EmotionDimension.VALENCE] == 0.9
        assert concept.primary_emotion.dimensions[EmotionDimension.AROUSAL] == 0.8

    def test_process_with_context(self, tmp_path: Path) -> None:
        """Test processing with emotional context."""
        db_manager, emotion_manager = _create_dependencies(tmp_path)
        db_manager.insert_or_update_word("work", "effort or activity", "noun")
        word_id = db_manager.get_word_id("work")
        emotion_manager.set_word_emotion(word_id, 0.0, 0.5)
        processor = _create_processor(db_manager, emotion_manager)

        context = processor.create_context_for_domain("academic")
        concept = processor.process_term("work", context)
        assert concept.term == "work"
        assert concept.word_id == word_id
        assert concept.primary_emotion.dimensions[EmotionDimension.VALENCE] != 0.0
