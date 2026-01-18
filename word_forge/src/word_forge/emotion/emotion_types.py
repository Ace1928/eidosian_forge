from __future__ import annotations

import datetime
import enum
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
)

# Type variables for generic emotion processing
E = TypeVar("E", bound="EmotionVector")  # Emotion type
C = TypeVar("C", bound="EmotionalContext")  # Context type


VADER_AVAILABLE = True
LLM_AVAILABLE = True

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE TYPE DEFINITIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class WordEmotionDict(TypedDict):
    """Type definition for word emotion data structure."""

    word_id: int
    valence: float
    arousal: float
    timestamp: float
    dominance: float  # Optional but supported in enhanced implementations


class MessageEmotionDict(TypedDict):
    """Type definition for message emotion data structure."""

    message_id: int
    label: str
    confidence: float
    timestamp: float


class EmotionAnalysisDict(TypedDict):
    """Type definition for emotion analysis results."""

    emotion_label: str
    confidence: float


class FullEmotionAnalysisDict(EmotionAnalysisDict, total=False):
    """Extended type definition including recursive emotion analysis."""

    concept: Dict[str, Any]
    dimensions: Dict[str, float]
    meta_emotions: List[Dict[str, Any]]
    patterns: Dict[str, List[Dict[str, Any]]]
    recursive_depth: int
    enhanced_insights: Dict[str, Any]
    emotional_entropy: float  # Measure of emotional complexity/uncertainty
    recursive_coherence: float  # Consistency across recursion levels


class VaderSentimentScores(TypedDict):
    """Type definition for VADER sentiment analyzer output."""

    neg: float  # Negative sentiment intensity
    neu: float  # Neutral sentiment intensity
    pos: float  # Positive sentiment intensity
    compound: float  # Normalized, weighted composite score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIMENSIONAL EMOTION MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EmotionDimension(enum.Enum):
    """Core emotional dimensions based on psychological models.

    Implements a comprehensive dimensional model of emotion with three category layers:
    1. Primary dimensions (PAD: valence, arousal, dominance)
    2. Extended dimensions (additional measurable aspects)
    3. Meta dimensions (recursive emotional qualities)

    This structure allows for representing both basic emotions and complex
    emotional states that include self-reference (emotions about emotions).
    """

    # Primary PAD dimensions (core of dimensional models)
    VALENCE = "valence"  # Positive vs. negative affect
    AROUSAL = "arousal"  # Intensity of emotional activation
    DOMINANCE = "dominance"  # Sense of control vs. submission

    # Extended psychological dimensions
    CERTAINTY = "certainty"  # Confidence in emotional assessment
    RELEVANCE = "relevance"  # Personal significance
    NOVELTY = "novelty"  # Unexpectedness or familiarity
    AGENCY = "agency"  # Attribution of causality
    SOCIAL = "social"  # Social connection or isolation
    TEMPORAL = "temporal"  # Time orientation (past/present/future)
    POTENCY = "potency"  # Strength/weakness dimension
    FAMILIARITY = "familiarity"  # Knownness/unknownness
    INTENSITY = "intensity"  # Raw strength of the emotion

    # Meta-emotional dimensions (emotions about emotions)
    META_COMPLEXITY = "meta_complexity"  # How complex the emotion is
    META_STABILITY = "meta_stability"  # How stable the emotion is over time
    META_CONGRUENCE = "meta_congruence"  # How congruent with other emotions
    META_CLARITY = "meta_clarity"  # Clarity of emotional awareness
    META_AWARENESS = "meta_awareness"  # Degree of emotional self-awareness

    @classmethod
    def primary_dimensions(cls) -> FrozenSet["EmotionDimension"]:
        """Return the three primary PAD model dimensions."""
        return frozenset({cls.VALENCE, cls.AROUSAL, cls.DOMINANCE})

    @classmethod
    def extended_dimensions(cls) -> FrozenSet["EmotionDimension"]:
        """Return extended PAD model dimensions."""
        return frozenset(
            cls.primary_dimensions()
            | {
                cls.POTENCY,
                cls.FAMILIARITY,
                cls.INTENSITY,
                cls.CERTAINTY,
                cls.RELEVANCE,
                cls.NOVELTY,
                cls.AGENCY,
                cls.SOCIAL,
                cls.TEMPORAL,
            }
        )

    @classmethod
    def meta_dimensions(cls) -> FrozenSet["EmotionDimension"]:
        """Return meta-emotional dimensions."""
        return frozenset(
            {
                cls.META_COMPLEXITY,
                cls.META_STABILITY,
                cls.META_CONGRUENCE,
                cls.META_CLARITY,
                cls.META_AWARENESS,
            }
        )

    @classmethod
    def all_dimensions(cls) -> FrozenSet["EmotionDimension"]:
        """Return all available emotional dimensions."""
        return frozenset(member for member in cls)

    @classmethod
    def get_dimension_category(cls, dimension: "EmotionDimension") -> str:
        """Determine the category a dimension belongs to."""
        if dimension in cls.primary_dimensions():
            return "primary"
        elif dimension in cls.extended_dimensions():
            return "extended"
        elif dimension in cls.meta_dimensions():
            return "meta"
        return "unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMOTION VECTOR SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass(frozen=True)
class EmotionVector:
    """Immutable n-dimensional vector representing emotional state.

    A mathematical representation of emotion as a point in multidimensional space,
    where each dimension corresponds to a specific emotional quality. The frozen
    design ensures emotional states are immutable, supporting pure functional
    operations that transform emotions without side effects.

    Attributes:
        dimensions: Dictionary mapping emotional dimensions to their values
        confidence: Certainty level about this emotional assessment (0.0-1.0)
    """

    dimensions: Dict[EmotionDimension, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate dimension values are within range."""
        # Ensure all dimension values are in valid range [-1.0, 1.0]
        for dim, value in self.dimensions.items():
            if not -1.0 <= value <= 1.0:
                object.__setattr__(
                    self,
                    "dimensions",
                    {**self.dimensions, dim: max(-1.0, min(1.0, value))},
                )

        # Ensure confidence is in valid range [0.0, 1.0]
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))

    def distance(self, other: "EmotionVector") -> float:
        """Calculate the n-dimensional Euclidean distance between emotion vectors."""
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        return math.sqrt(
            sum(
                (self.dimensions.get(dim, 0.0) - other.dimensions.get(dim, 0.0)) ** 2
                for dim in all_dims
            )
        )

    def blend(self, other: "EmotionVector", weight: float = 0.5) -> "EmotionVector":
        """Create a weighted blend of two emotion vectors.

        Args:
            other: The emotion vector to blend with
            weight: Weight of other vector (0.0 = all self, 1.0 = all other)

        Returns:
            A new vector representing the weighted average of both vectors
        """
        # Gather all dimensions from both vectors
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())

        # Create weighted combination of values
        new_dims = {
            dim: self.dimensions.get(dim, 0.0) * (1 - weight)
            + other.dimensions.get(dim, 0.0) * weight
            for dim in all_dims
        }

        # Blend confidence values too
        new_confidence = self.confidence * (1 - weight) + other.confidence * weight

        return EmotionVector(dimensions=new_dims, confidence=new_confidence)

    def inverse(self) -> "EmotionVector":
        """Return the emotional inverse (opposite in primary dimensions).

        Creates a vector with inverted values for primary dimensions,
        representing the emotional "opposite" of this state.

        Returns:
            New vector with primary dimensions inverted
        """
        return EmotionVector(
            dimensions={
                dim: -val
                for dim, val in self.dimensions.items()
                if dim in EmotionDimension.primary_dimensions()
            },
            confidence=self.confidence,
        )

    def intensify(self, factor: float = 1.5) -> "EmotionVector":
        """Intensify the emotion by the given factor.

        Amplifies the emotional state by scaling all dimension values.

        Args:
            factor: Scaling factor for intensity (default: 1.5)

        Returns:
            New vector with amplified emotional intensity
        """
        return EmotionVector(
            dimensions={dim: val * factor for dim, val in self.dimensions.items()},
            confidence=self.confidence,
        )

    def diminish(self, factor: float = 0.5) -> "EmotionVector":
        """Diminish the emotion by the given factor.

        Reduces the emotional state by scaling down all dimension values.

        Args:
            factor: Scaling factor for reduction (default: 0.5)

        Returns:
            New vector with reduced emotional intensity
        """
        return EmotionVector(
            dimensions={dim: val * factor for dim, val in self.dimensions.items()},
            confidence=self.confidence,
        )

    def normalized(self) -> "EmotionVector":
        """Return a normalized vector with magnitude 1.0.

        Creates a unit vector with the same direction but magnitude of 1.0,
        preserving the emotional "direction" while normalizing intensity.

        Returns:
            Normalized unit vector
        """
        magnitude_squared = sum(v * v for v in self.dimensions.values())
        if magnitude_squared == 0:
            return self

        magnitude = math.sqrt(magnitude_squared)
        return EmotionVector(
            dimensions={k: v / magnitude for k, v in self.dimensions.items()},
            confidence=self.confidence,
        )

    def dominant_dimension(self) -> Optional[Tuple[EmotionDimension, float]]:
        """Return the dimension with the largest absolute value.

        Identifies the most prominent emotional quality in this state.

        Returns:
            Tuple of (dimension, value) or None if no dimensions exist
        """
        if not self.dimensions:
            return None
        return max(self.dimensions.items(), key=lambda item: abs(item[1]))

    def contrast_with(self, other: "EmotionVector") -> "EmotionVector":
        """Create a vector highlighting differences between emotional states.

        Args:
            other: The emotion vector to contrast with

        Returns:
            Vector emphasizing dimensions where the two states differ most
        """
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        contrast_dims = {
            dim: self.dimensions.get(dim, 0.0) - other.dimensions.get(dim, 0.0)
            for dim in all_dims
        }
        # Filter out very small contrasts for clarity
        significant_contrasts = {
            dim: val for dim, val in contrast_dims.items() if abs(val) > 0.1
        }
        return EmotionVector(
            dimensions=significant_contrasts,
            confidence=min(self.confidence, other.confidence) * 0.9,
        )

    def resonate_with(self, other: "EmotionVector") -> float:
        """Calculate emotional resonance (similarity with directional emphasis).

        A more sophisticated similarity measure than simple distance,
        accounting for both alignment and intensity.

        Args:
            other: The emotion vector to measure resonance with

        Returns:
            Resonance value from 0.0 (none) to 1.0 (perfect)
        """
        # Get common dimensions
        common_dims = set(self.dimensions.keys()) & set(other.dimensions.keys())
        if not common_dims:
            return 0.0

        # Calculate dot product of common dimensions
        dot_product = sum(
            self.dimensions[dim] * other.dimensions[dim] for dim in common_dims
        )

        # Calculate magnitudes for common dimensions only
        self_magnitude = math.sqrt(
            sum(self.dimensions[dim] ** 2 for dim in common_dims)
        )
        other_magnitude = math.sqrt(
            sum(other.dimensions[dim] ** 2 for dim in common_dims)
        )

        # Avoid division by zero
        if self_magnitude == 0 or other_magnitude == 0:
            return 0.0

        # Calculate cosine similarity and rescale to [0,1] range
        cosine_similarity = dot_product / (self_magnitude * other_magnitude)
        return (cosine_similarity + 1) / 2

    def emotional_entropy(self) -> float:
        """Calculate the complexity/uncertainty of this emotional state.

        Higher entropy indicates a more complex, potentially ambiguous
        emotional state with multiple significant components.

        Returns:
            Entropy value from 0.0 (simple) to 1.0 (highly complex)
        """
        if not self.dimensions:
            return 0.0

        # Calculate normalized absolute values (like probabilities)
        total = sum(abs(v) for v in self.dimensions.values())
        if total == 0:
            return 0.0

        probs = [abs(v) / total for v in self.dimensions.values()]

        # Calculate Shannon entropy
        raw_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Normalize to [0,1] by dividing by max possible entropy
        max_entropy = math.log2(len(self.dimensions))
        return raw_entropy / max_entropy if max_entropy > 0 else 0.0

    def with_dimension(
        self, dimension: EmotionDimension, value: float
    ) -> "EmotionVector":
        """Create a new vector with an additional or updated dimension.

        Args:
            dimension: The emotional dimension to add or update
            value: The dimensional value (-1.0 to 1.0)

        Returns:
            New vector with the added/updated dimension
        """
        # Clamp value to valid range
        clamped_value = max(-1.0, min(1.0, value))

        # Create new dimensions dict with the added/updated dimension
        new_dimensions = {**self.dimensions, dimension: clamped_value}

        return EmotionVector(dimensions=new_dimensions, confidence=self.confidence)

    def filter_by_category(
        self, category: Literal["primary", "extended", "meta"]
    ) -> "EmotionVector":
        """Filter dimensions to include only those in the specified category.

        Args:
            category: Dimension category to keep

        Returns:
            New vector with only dimensions from the specified category
        """
        category_filter = {
            "primary": EmotionDimension.primary_dimensions(),
            "extended": EmotionDimension.extended_dimensions(),
            "meta": EmotionDimension.meta_dimensions(),
        }

        filtered_dims = {
            dim: val
            for dim, val in self.dimensions.items()
            if dim in category_filter.get(category, set[EmotionDimension]())
        }

        return EmotionVector(dimensions=filtered_dims, confidence=self.confidence)

    def as_dict(self) -> Dict[str, float]:
        """Convert to standard dictionary with string keys.

        Creates a serializable representation with string keys.

        Returns:
            Dictionary with dimension names as string keys
        """
        return {dim.value: val for dim, val in self.dimensions.items()}

    @classmethod
    def from_dict(
        cls, data: Dict[str, float], confidence: float = 1.0
    ) -> "EmotionVector":
        """Create an EmotionVector from a dictionary representation.

        Args:
            data: Dictionary with dimension names as keys and values as floats
            confidence: Overall confidence value for the vector

        Returns:
            New EmotionVector instance
        """
        # Convert string keys back to EmotionDimension enum values
        dimensions: Dict[EmotionDimension, float] = {}
        for key, value in data.items():
            try:
                # Try to find a matching enum value
                dim = EmotionDimension(key)
                dimensions[dim] = value
            except ValueError:
                # Skip invalid dimension names
                continue

        return cls(dimensions=dimensions, confidence=confidence)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMOTIONAL CONTEXT SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EmotionalContext:
    """Context that influences emotional processing.

    A framework for representing how emotions are modulated by various
    contextual factors including cultural, situational, temporal, and
    domain-specific influences.

    Attributes:
        cultural_factors: Cultural dimension adjustments
        situational_factors: Situation-specific adjustments
        temporal_factors: Time-related adjustments
        domain_specific: Domain-specific adjustments
        relationship_factors: Relational context adjustments
    """

    cultural_factors: Dict[str, float] = field(default_factory=dict)
    situational_factors: Dict[str, float] = field(default_factory=dict)
    temporal_factors: Dict[str, float] = field(default_factory=dict)
    domain_specific: Dict[str, float] = field(default_factory=dict)
    relationship_factors: Dict[str, float] = field(default_factory=dict)

    def apply_to_vector(self, vector: EmotionVector) -> EmotionVector:
        """Modify emotion vector based on contextual factors.

        Applies all contextual adjustments to an emotion vector, creating
        a new vector reflecting the emotion as modified by this context.

        Args:
            vector: The emotion vector to contextualize

        Returns:
            New vector with contextual adjustments applied
        """
        context_adjustment: Dict[EmotionDimension, float] = {}

        # Process each factor type using a consistent approach
        factor_sets = [
            self.cultural_factors,
            self.situational_factors,
            self.temporal_factors,
            self.domain_specific,
        ]

        # Apply all factor types
        for factor_set in factor_sets:
            for factor_name, factor_value in factor_set.items():
                try:
                    dim = EmotionDimension(factor_name)
                    context_adjustment[dim] = (
                        context_adjustment.get(dim, 0.0) + factor_value
                    )
                except ValueError:
                    # Skip invalid dimension names
                    continue

        # Create adjustment vector
        if not context_adjustment:
            return vector

        adjustment_vector = EmotionVector(
            dimensions=context_adjustment,
            confidence=0.8,  # Context application reduces confidence slightly
        )

        # Blend original vector with contextual adjustments
        return vector.blend(adjustment_vector, weight=0.3)

    def combine(
        self, other: "EmotionalContext", weight: float = 0.5
    ) -> "EmotionalContext":
        """Combine two contexts with weighted blending.

        Creates a new context that represents a weighted combination of two
        contexts, useful for merging different contextual frameworks.

        Args:
            other: Second context to combine with
            weight: Weight given to other context (0.0-1.0)

        Returns:
            New combined emotional context
        """
        result = EmotionalContext()

        # Combined dictionaries helper function
        def combine_dicts(
            dict1: Dict[str, float], dict2: Dict[str, float]
        ) -> Dict[str, float]:
            result = dict1.copy()
            for k, v in dict2.items():
                if k in result:
                    result[k] = result[k] * (1 - weight) + v * weight
                else:
                    result[k] = v * weight
            return result

        # Combine all factor dictionaries
        result.cultural_factors = combine_dicts(
            self.cultural_factors, other.cultural_factors
        )
        result.situational_factors = combine_dicts(
            self.situational_factors, other.situational_factors
        )
        result.temporal_factors = combine_dicts(
            self.temporal_factors, other.temporal_factors
        )
        result.domain_specific = combine_dicts(
            self.domain_specific, other.domain_specific
        )
        result.relationship_factors = combine_dicts(
            self.relationship_factors, other.relationship_factors
        )

        return result

    def dominance_factor(self) -> float:
        """Calculate how strongly this context influences emotions.

        Returns:
            A value from 0.0 (weak influence) to 1.0 (strong influence)
        """
        # Count the number of factors and their average magnitude
        all_factors = (
            list(self.cultural_factors.values())
            + list(self.situational_factors.values())
            + list(self.temporal_factors.values())
            + list(self.domain_specific.values())
            + list(self.relationship_factors.values())
        )

        if not all_factors:
            return 0.0

        # Consider both number of factors and their average magnitude
        factor_count = len(all_factors)
        avg_magnitude = sum(abs(v) for v in all_factors) / factor_count

        # Combine to get overall dominance (sigmoid to keep in range)
        raw_dominance = (factor_count / 10) * avg_magnitude * 2
        return min(raw_dominance, 1.0)

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to dictionary representation for serialization."""
        return {
            "cultural_factors": self.cultural_factors,
            "situational_factors": self.situational_factors,
            "temporal_factors": self.temporal_factors,
            "domain_specific": self.domain_specific,
            "relationship_factors": self.relationship_factors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, float]]) -> "EmotionalContext":
        """Create an EmotionalContext from a dictionary representation."""
        return cls(
            cultural_factors=data.get("cultural_factors", {}),
            situational_factors=data.get("situational_factors", {}),
            temporal_factors=data.get("temporal_factors", {}),
            domain_specific=data.get("domain_specific", {}),
            relationship_factors=data.get("relationship_factors", {}),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMOTIONAL CONCEPT SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class EmotionalConcept:
    """Full emotional representation of a lexical concept.

    A comprehensive model of how a term or concept relates to emotions,
    including primary emotions, secondary emotions, meta-emotions, and
    emotional patterns.

    This structure supports recursive emotional processing, where emotions
    can be analyzed, combined, and reflect upon themselves.

    Attributes:
        term: The lexical term being represented
        word_id: Database identifier for the term
        primary_emotion: The principal emotional vector
        secondary_emotions: Additional emotional associations
        meta_emotions: Emotions about emotions (recursive layer)
        emotional_patterns: Temporal or conditional emotional sequences
        relationship_context: How this concept relates emotionally to others
    """

    term: str
    word_id: int
    primary_emotion: EmotionVector
    secondary_emotions: List[Tuple[str, EmotionVector]] = field(default_factory=list)
    meta_emotions: List[Tuple[str, EmotionVector]] = field(default_factory=list)
    emotional_patterns: Dict[str, List[EmotionVector]] = field(default_factory=dict)
    relationship_context: Dict[str, List[Tuple[str, float]]] = field(
        default_factory=dict
    )

    @property
    def recursive_depth(self) -> int:
        """Determine the recursive emotional depth.

        Calculates how many layers of emotional recursion are present in this
        representation (emotions, emotions about emotions, etc.)

        Returns:
            Numeric depth value (higher means more recursive complexity)
        """
        return 1 + (1 if self.meta_emotions else 0) + len(self.emotional_patterns)

    def add_meta_emotion(self, label: str, emotion: EmotionVector) -> None:
        """Add an emotion about an emotion (meta-emotion).

        Adds a recursive emotional layer - an emotion felt about the primary
        emotion itself, representing emotional self-awareness.

        Args:
            label: Descriptive name for this meta-emotion
            emotion: The vector representing the meta-emotional state
        """
        self.meta_emotions.append((label, emotion))

    def add_emotional_pattern(
        self, pattern_type: str, sequence: List[EmotionVector]
    ) -> None:
        """Add a temporal or conditional pattern of emotions.

        Registers a sequence of emotional states that represent patterns
        like emotional transitions, responses, or conditional reactions.

        Args:
            pattern_type: Classification of the pattern (e.g., "transition", "response")
            sequence: Ordered list of emotion vectors in this pattern
        """
        self.emotional_patterns[pattern_type] = sequence

    def dominant_emotion(self) -> EmotionVector:
        """Return the most relevant emotion for this concept.

        Identifies the strongest emotional association based on arousal
        (emotional intensity), considering both primary and secondary emotions.

        Returns:
            The most dominant emotion vector
        """
        if not self.secondary_emotions:
            return self.primary_emotion

        # Return the strongest emotion based on intensity (arousal)
        all_emotions = [self.primary_emotion] + [e for _, e in self.secondary_emotions]
        return max(
            all_emotions,
            key=lambda e: abs(e.dimensions.get(EmotionDimension.AROUSAL, 0.0)),
        )

    def add_related_context(
        self, relationship_type: str, term: str, strength: float
    ) -> None:
        """Add a related term's contextual influence.

        Records how this concept emotionally relates to another term,
        building a network of emotional associations.

        Args:
            relationship_type: Type of relationship (e.g., "intensifies", "antonym")
            term: The related lexical term
            strength: Relationship strength (0.0-1.0)
        """
        if relationship_type not in self.relationship_context:
            self.relationship_context[relationship_type] = []
        self.relationship_context[relationship_type].append((term, strength))

    def emotional_coherence(self) -> float:
        """Calculate the internal consistency of emotional associations.

        Higher coherence indicates that primary, secondary, and meta emotions
        form a consistent emotional profile without significant contradictions.

        Returns:
            Coherence score from 0.0 (contradictory) to 1.0 (highly coherent)
        """
        if not self.secondary_emotions and not self.meta_emotions:
            return 1.0  # Only primary emotion exists, so trivially coherent

        # Collect all emotion vectors
        all_emotions = [self.primary_emotion] + [e for _, e in self.secondary_emotions]

        # Calculate average pairwise resonance between emotions
        if len(all_emotions) <= 1:
            primary_coherence = 1.0
        else:
            resonances: List[float] = []
            for i, emotion1 in enumerate(all_emotions):
                for emotion2 in all_emotions[i + 1 :]:
                    resonances.append(emotion1.resonate_with(emotion2))
            primary_coherence = sum(resonances) / len(resonances) if resonances else 1.0

        # If we have meta-emotions, evaluate their relationship to primary
        meta_coherence = 1.0
        if self.meta_emotions:
            meta_vectors = [e for _, e in self.meta_emotions]
            # Meta-emotions should have appropriate meta dimensions
            meta_dimension_presence = any(
                any(dim in EmotionDimension.meta_dimensions() for dim in e.dimensions)
                for e in meta_vectors
            )
            meta_coherence = 0.7 + (0.3 * int(meta_dimension_presence))

        # Combine both scores (primary emotions matter more)
        return primary_coherence * 0.7 + meta_coherence * 0.3

    def add_secondary_emotion(self, label: str, emotion: EmotionVector) -> None:
        """Add a secondary emotional association.

        Records an additional emotional association with a label, allowing
        for multiple emotional dimensions to be represented for this concept.

        Args:
            label: Descriptive name for this secondary emotion
            emotion: The vector representing the secondary emotional state
        """
        self.secondary_emotions.append((label, emotion))
        # Ensure the secondary emotion is normalized
        normalized_emotion = emotion.normalized()
        # Update the secondary emotion with the normalized version
        self.secondary_emotions[-1] = (label, normalized_emotion)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization.

        Creates a complete serializable representation of this concept
        for storage or transmission.

        Returns:
            Dictionary representation with all emotional components
        """
        return {
            "term": self.term,
            "word_id": self.word_id,
            "primary_emotion": {
                "dimensions": self.primary_emotion.as_dict(),
                "confidence": self.primary_emotion.confidence,
            },
            "secondary_emotions": [
                {
                    "label": label,
                    "dimensions": emotion.as_dict(),
                    "confidence": emotion.confidence,
                }
                for label, emotion in self.secondary_emotions
            ],
            "meta_emotions": [
                {
                    "label": label,
                    "dimensions": emotion.as_dict(),
                    "confidence": emotion.confidence,
                }
                for label, emotion in self.meta_emotions
            ],
            "emotional_patterns": {
                pattern_type: [
                    {"dimensions": emotion.as_dict(), "confidence": emotion.confidence}
                    for emotion in sequence
                ]
                for pattern_type, sequence in self.emotional_patterns.items()
            },
            "relationship_context": self.relationship_context,
            "recursive_depth": self.recursive_depth,
            "emotional_coherence": self.emotional_coherence(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalConcept":
        """Create an EmotionalConcept from a dictionary representation.

        Reconstructs a complete emotional concept from serialized data.

        Args:
            data: Dictionary representation from as_dict()

        Returns:
            Reconstructed EmotionalConcept instance
        """
        # Process primary emotion
        primary_data = data.get("primary_emotion", {})
        primary_dims = primary_data.get("dimensions", {})
        primary_conf = primary_data.get("confidence", 1.0)
        primary_emotion = EmotionVector.from_dict(primary_dims, primary_conf)

        # Process secondary emotions
        secondary_emotions: List[Tuple[str, EmotionVector]] = []
        for item in data.get("secondary_emotions", []):
            label = item.get("label", "unknown")
            dims = item.get("dimensions", {})
            conf = item.get("confidence", 1.0)
            emotion = EmotionVector.from_dict(dims, conf)
            secondary_emotions.append((label, emotion))

        # Process meta emotions
        meta_emotions: List[Tuple[str, EmotionVector]] = []
        for item in data.get("meta_emotions", []):
            label = item.get("label", "unknown")
            dims = item.get("dimensions", {})
            conf = item.get("confidence", 1.0)
            emotion = EmotionVector.from_dict(dims, conf)
            meta_emotions.append((label, emotion))

        # Process emotional patterns
        emotional_patterns: Dict[str, List[EmotionVector]] = {}
        for pattern_type, sequence_data in data.get("emotional_patterns", {}).items():
            sequence: List[EmotionVector] = []
            for item in sequence_data:
                dims = item.get("dimensions", {})
                conf = item.get("confidence", 1.0)
                emotion = EmotionVector.from_dict(dims, conf)
                sequence.append(emotion)
            emotional_patterns[pattern_type] = sequence

        return cls(
            term=data.get("term", ""),
            word_id=data.get("word_id", 0),
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            meta_emotions=meta_emotions,
            emotional_patterns=emotional_patterns,
            relationship_context=data.get("relationship_context", {}),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMOTION PROCESSOR PROTOCOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EmotionProcessor(Protocol):
    """Protocol defining the interface for emotion processors.

    A standard interface for components that process emotions, enabling
    modular, interchangeable emotional processing systems.
    """

    def process_term(
        self, term: str, context: Optional[EmotionalContext] = None
    ) -> EmotionalConcept: ...

    def analyze_relationship(
        self, term1: str, term2: str, relationship_type: str
    ) -> float: ...

    def get_emotion_vector(self, term: str) -> EmotionVector: ...


class EmotionTransformer(Protocol, Generic[E]):
    """Protocol for components that transform emotional states.

    Defines a standard interface for emotional transformation operations,
    enabling pipeline-based emotional processing.
    """

    def transform(self, emotion: E) -> E: ...
    def get_transformation_name(self) -> str: ...


class EmotionFactory(Protocol):
    """Protocol for components that create emotional vectors.

    Defines a standard interface for generating emotional representations
    from various input types.
    """

    def create_from_valence_arousal(
        self, valence: float, arousal: float, confidence: float = 1.0
    ) -> EmotionVector: ...

    def create_from_text(
        self, text: str, context: Optional[EmotionalContext] = None
    ) -> EmotionVector: ...

    def create_composite(
        self, emotions: List[Tuple[EmotionVector, float]]
    ) -> EmotionVector: ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMOTION ERROR HANDLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EmotionError(Exception):
    """Exception raised for errors in the emotion analysis subsystem.

    Used for all emotion-related errors to provide a consistent interface
    with contextual information about what went wrong during emotional
    processing operations.

    Attributes:
        message: Explanation of the error
        timestamp: When the error occurred
        context: Additional contextual information about the error
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize an emotion error with detailed context.

        Args:
            message: Descriptive error message
            context: Optional dictionary of contextual information
        """
        self.timestamp = datetime.datetime.now().isoformat()
        self.context = context or {}
        super().__init__(message)
