"""Tests for word_forge.emotion.emotion_types module.

This module tests the emotion type definitions including EmotionDimension,
EmotionVector, EmotionalContext, EmotionalConcept, and related functionality.
"""

import math

import pytest

from word_forge.emotion.emotion_types import (
    EmotionalContext,
    EmotionalConcept,
    EmotionDimension,
    EmotionError,
    EmotionVector,
)


class TestEmotionDimension:
    """Tests for the EmotionDimension enum."""

    def test_primary_dimensions(self):
        """Test that primary dimensions are returned."""
        primary = EmotionDimension.primary_dimensions()
        assert EmotionDimension.VALENCE in primary
        assert EmotionDimension.AROUSAL in primary
        assert EmotionDimension.DOMINANCE in primary
        assert len(primary) == 3

    def test_extended_dimensions(self):
        """Test that extended dimensions include primary."""
        extended = EmotionDimension.extended_dimensions()
        primary = EmotionDimension.primary_dimensions()
        assert primary <= extended

    def test_meta_dimensions(self):
        """Test that meta dimensions exist."""
        meta = EmotionDimension.meta_dimensions()
        assert EmotionDimension.META_COMPLEXITY in meta
        assert EmotionDimension.META_STABILITY in meta
        assert EmotionDimension.META_CONGRUENCE in meta

    def test_all_dimensions(self):
        """Test that all dimensions are returned."""
        all_dims = EmotionDimension.all_dimensions()
        assert len(all_dims) == len(EmotionDimension)

    def test_get_dimension_category_primary(self):
        """Test category identification for primary dimensions."""
        assert (
            EmotionDimension.get_dimension_category(EmotionDimension.VALENCE)
            == "primary"
        )

    def test_get_dimension_category_extended(self):
        """Test category identification for extended dimensions."""
        assert (
            EmotionDimension.get_dimension_category(EmotionDimension.CERTAINTY)
            == "extended"
        )

    def test_get_dimension_category_meta(self):
        """Test category identification for meta dimensions."""
        assert (
            EmotionDimension.get_dimension_category(EmotionDimension.META_COMPLEXITY)
            == "meta"
        )


class TestEmotionVector:
    """Tests for the EmotionVector dataclass."""

    def test_create_basic_vector(self):
        """Test creating a basic emotion vector."""
        vector = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.5, EmotionDimension.AROUSAL: 0.3}
        )
        assert vector.dimensions[EmotionDimension.VALENCE] == 0.5
        assert vector.dimensions[EmotionDimension.AROUSAL] == 0.3
        assert vector.confidence == 1.0

    def test_create_vector_with_confidence(self):
        """Test creating a vector with custom confidence."""
        vector = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.5}, confidence=0.8
        )
        assert vector.confidence == 0.8

    def test_value_clamping(self):
        """Test that values are clamped to valid range."""
        vector = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 2.0}  # Out of range
        )
        assert vector.dimensions[EmotionDimension.VALENCE] == 1.0

    def test_confidence_clamping(self):
        """Test that confidence is clamped to valid range."""
        vector = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.5}, confidence=1.5
        )
        assert vector.confidence == 1.0

    def test_distance_same_vector(self):
        """Test distance between identical vectors."""
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        assert vector.distance(vector) == 0.0

    def test_distance_different_vectors(self):
        """Test distance between different vectors."""
        vector1 = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.0})
        vector2 = EmotionVector(dimensions={EmotionDimension.VALENCE: 1.0})
        assert vector1.distance(vector2) == 1.0

    def test_blend_equal_weight(self):
        """Test blending with equal weights."""
        vector1 = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.0})
        vector2 = EmotionVector(dimensions={EmotionDimension.VALENCE: 1.0})
        blended = vector1.blend(vector2, weight=0.5)
        assert abs(blended.dimensions[EmotionDimension.VALENCE] - 0.5) < 0.001

    def test_blend_all_self(self):
        """Test blending with weight=0 (all self)."""
        vector1 = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.2})
        vector2 = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.8})
        blended = vector1.blend(vector2, weight=0.0)
        assert blended.dimensions[EmotionDimension.VALENCE] == 0.2

    def test_blend_all_other(self):
        """Test blending with weight=1 (all other)."""
        vector1 = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.2})
        vector2 = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.8})
        blended = vector1.blend(vector2, weight=1.0)
        assert blended.dimensions[EmotionDimension.VALENCE] == 0.8

    def test_inverse(self):
        """Test emotional inverse."""
        vector = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.5,
                EmotionDimension.AROUSAL: 0.3,
                EmotionDimension.DOMINANCE: -0.2,
            }
        )
        inverse = vector.inverse()
        assert inverse.dimensions[EmotionDimension.VALENCE] == -0.5
        assert inverse.dimensions[EmotionDimension.AROUSAL] == -0.3
        assert inverse.dimensions[EmotionDimension.DOMINANCE] == 0.2

    def test_intensify(self):
        """Test emotion intensification."""
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.4})
        intensified = vector.intensify(factor=2.0)
        assert intensified.dimensions[EmotionDimension.VALENCE] == 0.8

    def test_diminish(self):
        """Test emotion diminishing."""
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.8})
        diminished = vector.diminish(factor=0.5)
        assert diminished.dimensions[EmotionDimension.VALENCE] == 0.4

    def test_normalized_zero_magnitude(self):
        """Test normalized vector with zero magnitude."""
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.0})
        normalized = vector.normalized()
        assert normalized.dimensions[EmotionDimension.VALENCE] == 0.0

    def test_normalized_nonzero_magnitude(self):
        """Test normalized vector with non-zero magnitude."""
        vector = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.6, EmotionDimension.AROUSAL: 0.8}
        )
        normalized = vector.normalized()
        # Magnitude should be 1.0
        magnitude = math.sqrt(sum(v**2 for v in normalized.dimensions.values()))
        assert abs(magnitude - 1.0) < 0.001

    def test_dominant_dimension(self):
        """Test finding dominant dimension."""
        vector = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.3,
                EmotionDimension.AROUSAL: 0.8,
                EmotionDimension.DOMINANCE: 0.2,
            }
        )
        dominant = vector.dominant_dimension()
        assert dominant is not None
        assert dominant[0] == EmotionDimension.AROUSAL
        assert dominant[1] == 0.8

    def test_dominant_dimension_empty(self):
        """Test dominant dimension with empty vector."""
        vector = EmotionVector(dimensions={})
        assert vector.dominant_dimension() is None

    def test_contrast_with(self):
        """Test contrast between vectors."""
        vector1 = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.5, EmotionDimension.AROUSAL: 0.5}
        )
        vector2 = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.8, EmotionDimension.AROUSAL: 0.5}
        )
        contrast = vector1.contrast_with(vector2)
        # VALENCE should be significant, AROUSAL should be filtered
        assert EmotionDimension.VALENCE in contrast.dimensions
        assert EmotionDimension.AROUSAL not in contrast.dimensions

    def test_resonate_with(self):
        """Test resonance between similar vectors."""
        vector1 = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.5, EmotionDimension.AROUSAL: 0.5}
        )
        vector2 = EmotionVector(
            dimensions={EmotionDimension.VALENCE: 0.5, EmotionDimension.AROUSAL: 0.5}
        )
        resonance = vector1.resonate_with(vector2)
        assert abs(resonance - 1.0) < 0.0001  # Identical vectors have perfect resonance

    def test_resonate_with_no_common(self):
        """Test resonance with no common dimensions."""
        vector1 = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        vector2 = EmotionVector(dimensions={EmotionDimension.AROUSAL: 0.5})
        resonance = vector1.resonate_with(vector2)
        assert resonance == 0.0

    def test_emotional_entropy_simple(self):
        """Test entropy of a simple vector."""
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 1.0})
        entropy = vector.emotional_entropy()
        assert entropy == 0.0  # Single dimension has zero entropy

    def test_emotional_entropy_complex(self):
        """Test entropy of a complex vector."""
        vector = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.5,
                EmotionDimension.AROUSAL: 0.5,
            }
        )
        entropy = vector.emotional_entropy()
        assert 0.9 < entropy <= 1.0  # Equal values = max entropy

    def test_with_dimension(self):
        """Test adding a dimension."""
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        new_vector = vector.with_dimension(EmotionDimension.AROUSAL, 0.3)
        assert new_vector.dimensions[EmotionDimension.AROUSAL] == 0.3
        assert new_vector.dimensions[EmotionDimension.VALENCE] == 0.5

    def test_filter_by_category(self):
        """Test filtering by category."""
        vector = EmotionVector(
            dimensions={
                EmotionDimension.VALENCE: 0.5,
                EmotionDimension.META_COMPLEXITY: 0.3,
            }
        )
        filtered = vector.filter_by_category("primary")
        assert EmotionDimension.VALENCE in filtered.dimensions
        assert EmotionDimension.META_COMPLEXITY not in filtered.dimensions

    def test_as_dict(self):
        """Test conversion to dictionary."""
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        as_dict = vector.as_dict()
        assert "valence" in as_dict
        assert as_dict["valence"] == 0.5

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"valence": 0.5, "arousal": 0.3}
        vector = EmotionVector.from_dict(data, confidence=0.8)
        assert vector.dimensions[EmotionDimension.VALENCE] == 0.5
        assert vector.dimensions[EmotionDimension.AROUSAL] == 0.3
        assert vector.confidence == 0.8

    def test_from_dict_skips_invalid(self):
        """Test that from_dict skips invalid dimension names."""
        data = {"valence": 0.5, "invalid_dimension": 0.3}
        vector = EmotionVector.from_dict(data)
        assert EmotionDimension.VALENCE in vector.dimensions
        assert len(vector.dimensions) == 1


class TestEmotionalContext:
    """Tests for the EmotionalContext dataclass."""

    def test_default_context(self):
        """Test default context creation."""
        context = EmotionalContext()
        assert context.cultural_factors == {}
        assert context.situational_factors == {}
        assert context.temporal_factors == {}
        assert context.domain_specific == {}
        assert context.relationship_factors == {}

    def test_context_with_factors(self):
        """Test context with factors."""
        context = EmotionalContext(
            cultural_factors={"valence": 0.1},
            situational_factors={"arousal": 0.2},
        )
        assert context.cultural_factors["valence"] == 0.1
        assert context.situational_factors["arousal"] == 0.2

    def test_apply_to_vector_empty(self):
        """Test applying empty context to vector."""
        context = EmotionalContext()
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        result = context.apply_to_vector(vector)
        assert result.dimensions[EmotionDimension.VALENCE] == 0.5

    def test_apply_to_vector_with_factors(self):
        """Test applying context with factors to vector."""
        context = EmotionalContext(cultural_factors={"valence": 0.5})
        vector = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        result = context.apply_to_vector(vector)
        # Result should be blended
        assert result is not None

    def test_combine_contexts(self):
        """Test combining two contexts."""
        context1 = EmotionalContext(cultural_factors={"test": 1.0})
        context2 = EmotionalContext(cultural_factors={"test": 0.0})
        combined = context1.combine(context2, weight=0.5)
        assert combined.cultural_factors["test"] == 0.5

    def test_dominance_factor_empty(self):
        """Test dominance factor with no factors."""
        context = EmotionalContext()
        assert context.dominance_factor() == 0.0

    def test_dominance_factor_with_factors(self):
        """Test dominance factor with factors."""
        context = EmotionalContext(cultural_factors={"test": 0.5, "test2": 0.5})
        dominance = context.dominance_factor()
        assert dominance > 0.0

    def test_as_dict(self):
        """Test context conversion to dictionary."""
        context = EmotionalContext(cultural_factors={"test": 0.5})
        as_dict = context.as_dict()
        assert "cultural_factors" in as_dict
        assert as_dict["cultural_factors"]["test"] == 0.5

    def test_from_dict(self):
        """Test context creation from dictionary."""
        data = {
            "cultural_factors": {"test": 0.5},
            "situational_factors": {},
            "temporal_factors": {},
            "domain_specific": {},
            "relationship_factors": {},
        }
        context = EmotionalContext.from_dict(data)
        assert context.cultural_factors["test"] == 0.5


class TestEmotionalConcept:
    """Tests for the EmotionalConcept dataclass."""

    def test_create_concept(self):
        """Test creating a basic emotional concept."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(
            term="happy",
            word_id=1,
            primary_emotion=primary,
        )
        assert concept.term == "happy"
        assert concept.word_id == 1
        assert concept.primary_emotion == primary

    def test_recursive_depth_basic(self):
        """Test recursive depth for basic concept."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="happy", word_id=1, primary_emotion=primary)
        assert concept.recursive_depth == 1

    def test_recursive_depth_with_meta(self):
        """Test recursive depth with meta emotions."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="happy", word_id=1, primary_emotion=primary)
        concept.add_meta_emotion(
            "awareness",
            EmotionVector(dimensions={EmotionDimension.META_AWARENESS: 0.5}),
        )
        assert concept.recursive_depth == 2

    def test_add_meta_emotion(self):
        """Test adding meta emotion."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="happy", word_id=1, primary_emotion=primary)
        meta = EmotionVector(dimensions={EmotionDimension.META_CLARITY: 0.8})
        concept.add_meta_emotion("clarity", meta)
        assert len(concept.meta_emotions) == 1
        assert concept.meta_emotions[0][0] == "clarity"

    def test_add_emotional_pattern(self):
        """Test adding emotional pattern."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="happy", word_id=1, primary_emotion=primary)
        pattern = [
            EmotionVector(dimensions={EmotionDimension.VALENCE: 0.3}),
            EmotionVector(dimensions={EmotionDimension.VALENCE: 0.7}),
        ]
        concept.add_emotional_pattern("transition", pattern)
        assert "transition" in concept.emotional_patterns
        assert len(concept.emotional_patterns["transition"]) == 2

    def test_dominant_emotion_no_secondary(self):
        """Test dominant emotion with no secondary emotions."""
        primary = EmotionVector(dimensions={EmotionDimension.AROUSAL: 0.5})
        concept = EmotionalConcept(term="test", word_id=1, primary_emotion=primary)
        dominant = concept.dominant_emotion()
        assert dominant == primary

    def test_add_related_context(self):
        """Test adding related context."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="happy", word_id=1, primary_emotion=primary)
        concept.add_related_context("intensifies", "ecstatic", 0.8)
        assert "intensifies" in concept.relationship_context
        assert concept.relationship_context["intensifies"][0] == ("ecstatic", 0.8)

    def test_emotional_coherence_single_emotion(self):
        """Test coherence with single emotion."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="test", word_id=1, primary_emotion=primary)
        coherence = concept.emotional_coherence()
        assert coherence == 1.0

    def test_add_secondary_emotion(self):
        """Test adding secondary emotion."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="happy", word_id=1, primary_emotion=primary)
        secondary = EmotionVector(dimensions={EmotionDimension.AROUSAL: 0.6})
        concept.add_secondary_emotion("excitement", secondary)
        assert len(concept.secondary_emotions) == 1

    def test_as_dict(self):
        """Test concept conversion to dictionary."""
        primary = EmotionVector(dimensions={EmotionDimension.VALENCE: 0.5})
        concept = EmotionalConcept(term="happy", word_id=1, primary_emotion=primary)
        as_dict = concept.as_dict()
        assert as_dict["term"] == "happy"
        assert as_dict["word_id"] == 1
        assert "primary_emotion" in as_dict

    def test_from_dict(self):
        """Test concept creation from dictionary."""
        data = {
            "term": "happy",
            "word_id": 1,
            "primary_emotion": {
                "dimensions": {"valence": 0.5},
                "confidence": 1.0,
            },
            "secondary_emotions": [],
            "meta_emotions": [],
            "emotional_patterns": {},
            "relationship_context": {},
        }
        concept = EmotionalConcept.from_dict(data)
        assert concept.term == "happy"
        assert concept.word_id == 1


class TestEmotionError:
    """Tests for the EmotionError exception class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = EmotionError("Test error")
        assert str(error) == "Test error"
        assert error.timestamp is not None

    def test_error_with_context(self):
        """Test error with context."""
        error = EmotionError("Test error", context={"key": "value"})
        assert error.context == {"key": "value"}

    def test_error_is_exception(self):
        """Test that EmotionError is an Exception."""
        error = EmotionError("Test")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """Test that EmotionError can be raised."""
        with pytest.raises(EmotionError):
            raise EmotionError("Test error")
