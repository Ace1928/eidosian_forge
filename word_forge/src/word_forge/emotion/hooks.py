"""
Modular emotion processing hooks for the Word Forge emotion system.

This module provides reusable, functionally-isolated emotion processing components
that can be composed to create rich emotional representations. Each hook performs
a specific aspect of emotion analysis without side effects, following the principle
of single responsibility.

The hooks are organized into three categories:
1. Meta-emotion hooks: Add emotions about emotions
2. Pattern hooks: Add structured emotional patterns
3. Relationship hooks: Model emotional relationships between concepts

These hooks can be registered with the RecursiveEmotionProcessor to extend its
capabilities without modifying its core logic.
"""

from typing import Dict, List, Optional

from word_forge.emotion.emotion_types import (
    EmotionalConcept,
    EmotionDimension,
    EmotionVector,
)

# ==========================================
# Meta-Emotion Hooks
# ==========================================


def add_clarity_meta_emotion(concept: EmotionalConcept) -> None:
    """
    Add meta-emotions related to emotional clarity/awareness.

    Adds an awareness meta-emotion when the primary emotion has strong valence,
    representing the subject's awareness of their clear emotional state.

    Args:
        concept: The emotional concept to enhance with meta-emotions
    """
    primary = concept.primary_emotion
    valence = primary.dimensions.get(EmotionDimension.VALENCE, 0.0)

    if abs(valence) > 0.7:  # Strong valence
        meta_dimensions = {
            EmotionDimension.VALENCE: 0.3,  # Slightly positive about having clear emotion
            EmotionDimension.AROUSAL: 0.2,  # Low arousal meta-emotion
            EmotionDimension.CERTAINTY: 0.8,  # High certainty about this evaluation
            EmotionDimension.META_CLARITY: 0.9,  # High clarity in emotional awareness
        }
        concept.add_meta_emotion(
            "awareness_of_emotional_clarity",
            EmotionVector(dimensions=meta_dimensions, confidence=0.8),
        )


def add_arousal_meta_emotion(concept: EmotionalConcept) -> None:
    """
    Add meta-emotions related to emotional arousal/activation.

    When primary emotion has high arousal, adds a meta-emotion representing
    the response to this activation (which may be positive for moderate
    arousal or negative for extreme arousal).

    Args:
        concept: The emotional concept to enhance with meta-emotions
    """
    primary = concept.primary_emotion
    arousal = primary.dimensions.get(EmotionDimension.AROUSAL, 0.0)

    if arousal > 0.7:  # High arousal
        meta_dimensions = {
            EmotionDimension.VALENCE: -0.2 if arousal > 0.9 else 0.2,
            EmotionDimension.AROUSAL: -0.5,  # Low arousal response to high arousal
            EmotionDimension.DOMINANCE: 0.3 if arousal < 0.9 else -0.3,
            EmotionDimension.META_STABILITY: -0.4,  # Recognition of instability
        }
        concept.add_meta_emotion(
            "response_to_activation",
            EmotionVector(dimensions=meta_dimensions, confidence=0.7),
        )


def add_complexity_meta_emotion(concept: EmotionalConcept) -> None:
    """
    Add meta-emotions related to emotional complexity.

    When the primary emotion contains multiple significant dimensions,
    adds a meta-emotion representing awareness of this complexity.

    Args:
        concept: The emotional concept to enhance with meta-emotions
    """
    primary = concept.primary_emotion
    dimensions_count = len([v for v in primary.dimensions.values() if abs(v) > 0.5])

    if dimensions_count >= 3:  # Complex emotion with multiple dimensions
        meta_dimensions = {
            EmotionDimension.VALENCE: 0.2,  # Slightly positive about complexity
            EmotionDimension.AROUSAL: 0.3,  # Moderate arousal
            EmotionDimension.CERTAINTY: -0.3,  # Lower certainty due to complexity
            EmotionDimension.META_COMPLEXITY: 0.8,  # High awareness of complexity
        }
        concept.add_meta_emotion(
            "awareness_of_emotional_complexity",
            EmotionVector(dimensions=meta_dimensions, confidence=0.7),
        )


def add_congruence_meta_emotion(concept: EmotionalConcept) -> None:
    """
    Add meta-emotions related to emotional congruence.

    For concepts with secondary emotions, adds a meta-emotion representing
    the coherence or conflict between primary and secondary emotions.

    Args:
        concept: The emotional concept to enhance with meta-emotions
    """
    # Only applicable for concepts with secondary emotions
    if not concept.secondary_emotions:
        return

    primary = concept.primary_emotion
    primary_valence = primary.dimensions.get(EmotionDimension.VALENCE, 0.0)

    # Calculate average valence of secondary emotions
    sec_valence_sum = 0.0
    sec_count = 0

    for _, sec_emotion in concept.secondary_emotions:
        sec_valence = sec_emotion.dimensions.get(EmotionDimension.VALENCE, 0.0)
        sec_valence_sum += sec_valence
        sec_count += 1

    if sec_count == 0:
        return

    avg_sec_valence = sec_valence_sum / sec_count

    # Calculate congruence (similarity in valence direction)
    congruence = primary_valence * avg_sec_valence

    if congruence > 0.2:  # Congruent emotions
        meta_dimensions = {
            EmotionDimension.VALENCE: 0.4,  # Positive about congruence
            EmotionDimension.AROUSAL: -0.2,  # Lower arousal from harmony
            EmotionDimension.CERTAINTY: 0.6,  # Higher certainty due to consistency
            EmotionDimension.META_CONGRUENCE: 0.8,  # High congruence
        }
        concept.add_meta_emotion(
            "emotional_harmony",
            EmotionVector(dimensions=meta_dimensions, confidence=0.7),
        )
    elif congruence < -0.2:  # Conflicting emotions
        meta_dimensions = {
            EmotionDimension.VALENCE: -0.3,  # Negative about conflict
            EmotionDimension.AROUSAL: 0.4,  # Higher arousal from dissonance
            EmotionDimension.CERTAINTY: -0.5,  # Lower certainty due to conflict
            EmotionDimension.META_CONGRUENCE: -0.7,  # Low congruence
        }
        concept.add_meta_emotion(
            "emotional_dissonance",
            EmotionVector(dimensions=meta_dimensions, confidence=0.7),
        )


# ==========================================
# Emotional Pattern Hooks
# ==========================================


def add_temporal_sequence(concept: EmotionalConcept) -> None:
    """
    Add temporal sequence pattern (how the emotion evolves over time).

    Creates a three-phase emotional arc consisting of onset, peak, and offset
    phases, modeling the typical evolution of an emotional experience.

    Args:
        concept: The emotional concept to enhance with patterns
    """
    primary = concept.primary_emotion
    timeline: List[EmotionVector] = []

    # Starting emotion (onset)
    start_emotion = primary.diminish(0.7)
    timeline.append(start_emotion)

    # Peak emotion
    timeline.append(primary)

    # Declining emotion (offset)
    end_emotion = primary.diminish(0.5)
    # Shift valence slightly toward neutral
    end_dims = dict(end_emotion.dimensions)
    if EmotionDimension.VALENCE in end_dims:
        end_dims[EmotionDimension.VALENCE] *= 0.7  # Reduce valence intensity
    timeline.append(
        EmotionVector(dimensions=end_dims, confidence=end_emotion.confidence)
    )

    concept.add_emotional_pattern("temporal_sequence", timeline)


def add_intensity_gradation(concept: EmotionalConcept) -> None:
    """
    Add intensity gradation pattern (emotion at different intensity levels).

    Creates a five-point intensity scale from minimal to extreme, modeling
    how the emotion manifests at different intensity levels.

    Args:
        concept: The emotional concept to enhance with patterns
    """
    primary = concept.primary_emotion
    gradation: List[EmotionVector] = []

    # Minimal intensity (threshold of perception)
    minimal = EmotionVector(
        dimensions={dim: val * 0.2 for dim, val in primary.dimensions.items()},
        confidence=0.7,
    )
    gradation.append(minimal)

    # Low intensity
    low = EmotionVector(
        dimensions={dim: val * 0.5 for dim, val in primary.dimensions.items()},
        confidence=0.8,
    )
    gradation.append(low)

    # Medium intensity (standard)
    gradation.append(primary)

    # High intensity
    high = EmotionVector(
        dimensions={
            dim: max(-1.0, min(1.0, val * 1.5))
            for dim, val in primary.dimensions.items()
        },
        confidence=0.8,
    )
    gradation.append(high)

    # Extreme intensity (may have different qualities)
    extreme_dims = {
        dim: max(-1.0, min(1.0, val * 2.0)) for dim, val in primary.dimensions.items()
    }

    # Extreme intensity often has different emotional qualities
    # Add some complexity/instability for extreme emotions
    if EmotionDimension.CERTAINTY not in extreme_dims:
        extreme_dims[EmotionDimension.CERTAINTY] = -0.4  # Less certainty at extremes
    if EmotionDimension.SOCIAL not in extreme_dims:
        extreme_dims[EmotionDimension.SOCIAL] = -0.3  # Often more isolating at extremes

    extreme = EmotionVector(dimensions=extreme_dims, confidence=0.7)
    gradation.append(extreme)

    concept.add_emotional_pattern("intensity_gradation", gradation)


def add_contextual_variations(concept: EmotionalConcept) -> None:
    """
    Add contextual variations pattern (emotion in different contexts).

    Models how the emotion would manifest differently in personal,
    professional, social, and cultural contexts.

    Args:
        concept: The emotional concept to enhance with patterns
    """
    primary = concept.primary_emotion
    variations: List[EmotionVector] = []

    # Personal/intimate context
    personal_dims = dict(primary.dimensions)
    personal_dims[EmotionDimension.INTENSITY] = min(
        1.0, personal_dims.get(EmotionDimension.INTENSITY, 0) + 0.3
    )
    personal_dims[EmotionDimension.RELEVANCE] = min(
        1.0, personal_dims.get(EmotionDimension.RELEVANCE, 0) + 0.4
    )
    personal_dims[EmotionDimension.SOCIAL] = min(
        1.0, personal_dims.get(EmotionDimension.SOCIAL, 0) + 0.5
    )
    variations.append(EmotionVector(dimensions=personal_dims, confidence=0.7))

    # Professional context
    professional_dims = dict(primary.dimensions)
    professional_dims[EmotionDimension.INTENSITY] = max(
        -1.0, professional_dims.get(EmotionDimension.INTENSITY, 0) - 0.3
    )
    professional_dims[EmotionDimension.DOMINANCE] = min(
        1.0, professional_dims.get(EmotionDimension.DOMINANCE, 0) + 0.2
    )
    professional_dims[EmotionDimension.CERTAINTY] = min(
        1.0, professional_dims.get(EmotionDimension.CERTAINTY, 0) + 0.3
    )
    variations.append(EmotionVector(dimensions=professional_dims, confidence=0.7))

    # Social context
    social_dims = dict(primary.dimensions)
    if (
        EmotionDimension.VALENCE in social_dims
        and social_dims[EmotionDimension.VALENCE] < 0
    ):
        # Negative emotions often moderated in social contexts
        social_dims[EmotionDimension.VALENCE] *= 0.7
        social_dims[EmotionDimension.INTENSITY] = max(
            -1.0, social_dims.get(EmotionDimension.INTENSITY, 0) - 0.2
        )
    social_dims[EmotionDimension.SOCIAL] = min(
        1.0, social_dims.get(EmotionDimension.SOCIAL, 0) + 0.6
    )
    variations.append(EmotionVector(dimensions=social_dims, confidence=0.7))

    # Cultural context
    cultural_dims = dict(primary.dimensions)
    # Cultural contexts often modify emotional display rules
    cultural_dims[EmotionDimension.INTENSITY] = max(
        -1.0, cultural_dims.get(EmotionDimension.INTENSITY, 0) - 0.2
    )
    cultural_dims[EmotionDimension.CERTAINTY] = max(
        -1.0, cultural_dims.get(EmotionDimension.CERTAINTY, 0) - 0.3
    )
    variations.append(EmotionVector(dimensions=cultural_dims, confidence=0.6))

    concept.add_emotional_pattern("contextual_variations", variations)


# ==========================================
# Secondary Emotion and Relationship Hooks
# ==========================================


def add_secondary_emotions(concept: EmotionalConcept) -> None:
    """
    Add secondary emotions based on related terms.

    This is a stub implementation that requires processor's internal state.
    In practice, the emotion processor would pass related terms to this hook
    through a closure or provide the DB access needed to retrieve them.

    Args:
        concept: The emotional concept to enhance with secondary emotions
    """
    # This hook requires access to related terms, which typically comes from
    # the processor's database. This implementation is a placeholder that
    # should be initialized with proper data sources by the processor.
    pass


def add_emotional_patterns(concept: EmotionalConcept) -> None:
    """
    Add all standard emotional patterns to a concept.

    A convenience function that applies all basic pattern hooks
    to add complete emotional pattern information.

    Args:
        concept: The emotional concept to enhance with patterns
    """
    # Apply all pattern hooks in sequence
    add_temporal_sequence(concept)
    add_intensity_gradation(concept)
    add_contextual_variations(concept)


def add_relationship_context(
    concept: EmotionalConcept,
    related_concepts: Optional[Dict[str, EmotionalConcept]] = None,
) -> None:
    """
    Add relationship context between this emotion and related emotions.

    Enriches the concept with information about how it relates to
    other emotional concepts in a semantic network.

    Args:
        concept: The emotional concept to enhance with relationship context
        related_concepts: Dictionary of related concepts keyed by relationship type
    """
    if not related_concepts:
        return

    # Process each related concept by relationship type
    for rel_type, rel_concept in related_concepts.items():
        primary = concept.primary_emotion
        related = rel_concept.primary_emotion

        # Calculate relationship strength based on relationship type
        strength = 0.0

        if rel_type in ("intensifies", "amplifies"):
            # Check if related emotion is an intensified version
            intensity_ratio = _calculate_intensity_ratio(related, primary)
            if intensity_ratio > 1.2:
                strength = min(1.0, (intensity_ratio - 1.0) * 2)

        elif rel_type in ("precedes", "follows"):
            # Temporal relationship - common in emotional sequences
            temporal_link = _calculate_temporal_relationship(primary, related)
            strength = max(0.0, temporal_link)

        elif rel_type in ("enables", "inhibits"):
            # One emotion affecting the likelihood of another
            valence_product = primary.dimensions.get(
                EmotionDimension.VALENCE, 0
            ) * related.dimensions.get(EmotionDimension.VALENCE, 0)
            strength = 0.5 + (valence_product * 0.5)

        if strength > 0.3:
            # Add relationship information to the concept
            concept.add_related_context(rel_type, rel_concept.term, strength)


# ==========================================
# Helper Functions
# ==========================================


def _calculate_intensity_ratio(
    emotion1: EmotionVector, emotion2: EmotionVector
) -> float:
    """Calculate the intensity ratio between two emotions."""
    intensity1 = sum(abs(v) for v in emotion1.dimensions.values()) / max(
        1, len(emotion1.dimensions)
    )
    intensity2 = sum(abs(v) for v in emotion2.dimensions.values()) / max(
        1, len(emotion2.dimensions)
    )
    return intensity1 / max(0.0001, intensity2)  # Avoid division by zero


def _calculate_temporal_relationship(
    emotion1: EmotionVector, emotion2: EmotionVector
) -> float:
    """Calculate likelihood of temporal relationship between emotions."""
    # Check for arousal pattern typical of emotional sequences
    arousal1 = emotion1.dimensions.get(EmotionDimension.AROUSAL, 0)
    arousal2 = emotion2.dimensions.get(EmotionDimension.AROUSAL, 0)

    # Check for valence shift typical of emotional progressions
    valence1 = emotion1.dimensions.get(EmotionDimension.VALENCE, 0)
    valence2 = emotion2.dimensions.get(EmotionDimension.VALENCE, 0)

    # Emotional sequences often involve arousal changes
    arousal_change = abs(arousal2 - arousal1)

    # And sometimes involve valence resolution (e.g., negative to positive)
    valence_resolution = valence1 < 0 and valence2 > 0

    # Combine factors
    temporal_score = arousal_change * 0.6
    if valence_resolution:
        temporal_score += 0.4

    return min(1.0, temporal_score)
