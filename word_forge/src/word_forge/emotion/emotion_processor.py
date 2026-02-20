import hashlib
import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, final

from eidosian_core import eidosian

from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.emotion.emotion_types import (
    EmotionalConcept,
    EmotionalContext,
    EmotionDimension,
    EmotionVector,
)
from word_forge.emotion.hooks import (
    add_arousal_meta_emotion,
    add_clarity_meta_emotion,
    add_complexity_meta_emotion,
    add_congruence_meta_emotion,
    add_contextual_variations,
    add_intensity_gradation,
    add_temporal_sequence,
)

# Configure logger
logger = logging.getLogger(__name__)


@final
class RecursiveEmotionProcessor:
    """
    Advanced emotion processor that analyzes lexical constructs through multiple
    dimensions and recursive relationships of emotion.

    This processor maps words and phrases to comprehensive emotion representations
    that capture nuance, context, and meta-emotional qualities.
    """

    def __init__(self, db_manager: DBManager, emotion_manager: EmotionManager):
        """
        Initialize the emotion processor with necessary dependencies.

        Args:
            db_manager: Database manager for accessing word data
            emotion_manager: Manager for storing and retrieving emotional attributes
        """
        self.db = db_manager
        self.emotion_manager = emotion_manager
        self.context_registry: Dict[str, EmotionalContext] = {}
        self._cache: Dict[str, EmotionalConcept] = {}
        self._processing_depth = 0
        self._max_recursion = 3
        # We'll properly type this in _initialize_nlp when spaCy is imported
        self._reference_docs = {}  # Store spaCy doc objects
        # Add vector cache
        self._vector_cache: Dict[str, Any] = {}

        # Initialize NLP to None - will be set in _initialize_nlp() if spaCy is available
        self.nlp: Any = None

        # Initialize hook registries
        self.meta_emotion_hooks: List[Callable[[EmotionalConcept], None]] = []
        self.pattern_hooks: List[Callable[[EmotionalConcept], None]] = []

        # Hooks and Meta Emotion Hooks

        # Register built-in hooks
        self._register_default_hooks()

        # Initialize NLP components if available
        self._initialize_nlp()

    def _initialize_nlp(self) -> None:
        """Initialize NLP components for emotion analysis."""
        try:
            import spacy

            # Use the smaller model for resource optimised semantic analysis
            self.nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            print("Warning: spaCy 'en_core_web_sm' model not available. Using fallback methods.")
            self.nlp = None

    @contextmanager
    def _recursive_scope(self):
        """Context manager for tracking recursion depth.

        Ensures processing depth is properly incremented and decremented,
        even if exceptions occur during processing.

        Yields:
            None: Control returns to the calling context
        """
        self._processing_depth += 1
        try:
            yield
        finally:
            self._processing_depth -= 1

    @eidosian()
    def process_term(self, term: str, context: Optional[EmotionalContext] = None) -> EmotionalConcept:
        """
        Process a term to extract its full emotional representation.

        Args:
            term: The word or phrase to process
            context: Optional emotional context to consider

        Returns:
            Complete emotional concept with primary, secondary and meta emotions
        """
        # Check cache first
        cache_key = self._get_cache_key(term, context)
        if cache_key in self._cache:
            # If we have context but cached version doesn't, apply context
            cached = self._cache[cache_key]
            if context:
                # Create a copy with context applied
                new_primary = context.apply_to_vector(cached.primary_emotion)
                new_concept = EmotionalConcept(
                    term=cached.term,
                    word_id=cached.word_id,
                    primary_emotion=new_primary,
                    secondary_emotions=cached.secondary_emotions,
                    meta_emotions=cached.meta_emotions,
                    emotional_patterns=cached.emotional_patterns,
                    relationship_context=cached.relationship_context,
                )
                return new_concept
            return cached

        # Prevent excessive recursion
        with self._recursive_scope():
            if self._processing_depth > self._max_recursion:
                # Return simplified emotion when recursion limit reached
                return self._create_fallback_concept(term)

            try:
                # Get word ID from database
                word_id = self._get_word_id(term)
                if not word_id:
                    # Handle unknown words by creating a new entry
                    word_id = self._create_new_word_entry(term)

                # Get base emotional vector
                emotion_vector = self._extract_base_emotion(word_id, term)

                # Apply contextual factors if provided
                if context:
                    emotion_vector = context.apply_to_vector(emotion_vector)

                # Create the emotional concept
                concept = EmotionalConcept(term=term, word_id=word_id, primary_emotion=emotion_vector)

                # Add secondary emotions based on relationships
                self._add_secondary_emotions(concept)

                # Add meta-emotions (emotions about emotions)
                self._add_meta_emotions(concept)

                # Add emotional patterns
                self._add_emotional_patterns(concept)

                # Cache the result
                self._cache[cache_key] = concept
                return concept

            finally:
                self._processing_depth -= 1

    @eidosian()
    def analyze_relationship(self, term1: str, term2: str, relationship_type: str) -> float:
        """
        Analyze the emotional relationship between two terms.

        Args:
            term1: First term
            term2: Second term
            relationship_type: Type of relationship to analyze

        Returns:
            Strength of the emotional relationship (0.0-1.0)
        """
        # Get emotion vectors for both terms
        emotion1 = self.get_emotion_vector(term1)
        emotion2 = self.get_emotion_vector(term2)

        # Calculate relationship strength based on type
        if relationship_type in ("emotional_synonym", "synonym"):
            # Emotional synonyms should have similar emotional vectors
            distance = emotion1.distance(emotion2)
            return max(0.0, 1.0 - (distance / 2.0))

        elif relationship_type in ("emotional_antonym", "antonym"):
            # Emotional antonyms should have opposite primary dimensions
            inverse = emotion1.inverse()
            distance = inverse.distance(emotion2)
            return max(0.0, 1.0 - (distance / 2.0))

        elif relationship_type == "intensifies":
            # Check if second term intensifies the first
            intensified = emotion1.intensify()
            distance = intensified.distance(emotion2)
            return max(0.0, 1.0 - (distance / 2.0))

        elif relationship_type == "diminishes":
            # Check if second term diminishes the first
            diminished = emotion1.diminish()
            distance = diminished.distance(emotion2)
            return max(0.0, 1.0 - (distance / 2.0))

        elif relationship_type in ("evokes", "responds_to"):
            # These relationships don't require similarity, just connection
            # Check if dimensions have expected correlations
            return self._calculate_evocative_strength(emotion1, emotion2)

        elif relationship_type.endswith("_related"):
            # Check similarity in the specific dimension
            dim_name = relationship_type.split("_")[0].upper()
            try:
                dimension = EmotionDimension[dim_name]
                val1 = emotion1.dimensions.get(dimension, 0.0)
                val2 = emotion2.dimensions.get(dimension, 0.0)
                return 1.0 - abs(val1 - val2)
            except KeyError:
                return 0.0

        elif relationship_type in ("emotional_component", "emotional_composite"):
            # Components should be simpler emotions that make up composites
            # Check if term1 is simpler (fewer dimensions with strong values)
            if relationship_type == "emotional_component":
                return self._calculate_component_strength(emotion1, emotion2)
            else:
                return self._calculate_component_strength(emotion2, emotion1)

        elif relationship_type == "meta_emotion":
            # Meta-emotions are about other emotions
            # See if term2 has meta-emotional qualities (certainty, complexity)
            meta_dims = {d: emotion2.dimensions.get(d, 0.0) for d in EmotionDimension.meta_dimensions()}
            if not meta_dims:
                return 0.0

            # Calculate average strength of meta dimensions
            return sum(abs(v) for v in meta_dims.values()) / len(meta_dims)

        else:
            # Default calculation based on emotional distance
            distance = emotion1.distance(emotion2)
            return max(0.0, 1.0 - (distance / 2.0))

    @eidosian()
    def get_emotion_vector(self, term: str) -> EmotionVector:
        """
        Get the primary emotion vector for a term.

        Args:
            term: Term to analyze

        Returns:
            Primary emotion vector for the term
        """
        # Check if we've already processed this term
        if term in self._cache:
            return self._cache[term].primary_emotion

        # Get word ID
        word_id = self._get_word_id(term)
        if not word_id:
            # Create default emotion for unknown terms
            return self._create_default_emotion(term)

        # Extract emotion from database
        return self._extract_base_emotion(word_id, term)

    @eidosian()
    def register_context(self, name: str, context: EmotionalContext) -> None:
        """
        Register a named emotional context for reuse.

        Args:
            name: Unique name for this context
            context: The emotional context to register
        """
        self.context_registry[name] = context

    @eidosian()
    def get_context(self, name: str) -> Optional[EmotionalContext]:
        """
        Retrieve a registered emotional context.

        Args:
            name: Name of the context to retrieve

        Returns:
            The registered context or None if not found
        """
        return self.context_registry.get(name)

    @eidosian()
    def create_context_for_domain(self, domain: str) -> EmotionalContext:
        """
        Create a context appropriate for a specific domain.

        Args:
            domain: The domain name (e.g., "academic", "casual", "medical")

        Returns:
            A domain-specific emotional context
        """
        context = EmotionalContext()

        if domain == "academic":
            context.domain_specific = {
                "valence": 0.1,  # Slightly positive
                "arousal": -0.4,  # Lower arousal
                "dominance": 0.3,  # Higher dominance
                "certainty": 0.7,  # Higher certainty
                "social": -0.2,  # Less social
            }
        elif domain == "casual":
            context.domain_specific = {
                "valence": 0.3,  # More positive
                "arousal": 0.4,  # Higher arousal
                "dominance": 0.0,  # Neutral dominance
                "certainty": -0.2,  # Lower certainty
                "social": 0.5,  # More social
            }
        elif domain == "medical":
            context.domain_specific = {
                "valence": -0.2,  # Slightly negative
                "arousal": -0.1,  # Slightly lower arousal
                "dominance": 0.5,  # Higher dominance
                "certainty": 0.8,  # High certainty
                "relevance": 0.9,  # High relevance
            }
        elif domain == "literary":
            context.domain_specific = {
                "valence": 0.0,  # Neutral valence
                "arousal": 0.2,  # Slightly higher arousal
                "dominance": 0.0,  # Neutral dominance
                "novelty": 0.7,  # High novelty
                "temporal": 0.5,  # Higher temporal focus
            }

        return context

    def _get_word_id(self, term: str) -> Optional[int]:
        """Get the database ID for a word."""
        try:
            return self.db.get_word_id(term)
        except Exception:
            return None

    def _create_new_word_entry(self, term: str) -> int:
        """Create a new word entry in the database."""
        try:
            self.db.insert_or_update_word(
                term=term,
                definition=f"Auto-generated entry for '{term}'",
                part_of_speech="unknown",  # Using part_of_speech instead of pos
            )
            # Get the word ID after insertion
            return self.db.get_word_id(term) or -1
        except Exception as e:
            print(f"Error creating word entry: {e}")
            return -1

    def _extract_base_emotion(self, word_id: int, term: str) -> EmotionVector:
        """Extract base emotion from database or generate if not present."""
        # Try to get existing emotion
        emotion_data = self.emotion_manager.get_word_emotion(word_id)

        if emotion_data:
            # Use existing data to create vector
            dimensions = {
                EmotionDimension.VALENCE: emotion_data.get("valence", 0.0),
                EmotionDimension.AROUSAL: emotion_data.get("arousal", 0.0),
            }

            # Add dominance if available
            if "dominance" in emotion_data:
                dimensions[EmotionDimension.DOMINANCE] = emotion_data["dominance"]

            return EmotionVector(dimensions=dimensions)
        else:
            # Generate new emotion vector
            vector = self._generate_emotion_for_term(term)

            # Feedback loop: store high-confidence vectors back to the database
            if vector.confidence >= 0.7 and word_id > 0:
                self.emotion_manager.set_word_emotion(
                    word_id,
                    vector.dimensions.get(EmotionDimension.VALENCE, 0.0),
                    vector.dimensions.get(EmotionDimension.AROUSAL, 0.0),
                )

            return vector

    def _generate_emotion_for_term(self, term: str) -> EmotionVector:
        """
        Generate emotion for a term using optimal available method.

        Implements a cascading strategy with graceful degradation:
        1. Vector-based generation (highest accuracy)
        2. LLM-based generation (high contextual understanding)
        3. Heuristic generation (fallback reliability)
        """
        # Try vector-based generation first
        vector_emotion = self._generate_vector_based_emotion(term)

        # Try LLM-based generation as second option
        llm_emotion = self._generate_llm_based_emotion(term)

        # Fall back to heuristic generation
        heuristic_emotion = self._generate_heuristic_emotion(term)

        # Integrate results of all emotion generation approaches with dynamic weighting
        emotions: List[EmotionVector] = []
        confidences: List[float] = []

        # Collect available emotions with their confidence values
        if vector_emotion and vector_emotion.confidence > 0:
            emotions.append(vector_emotion)
            confidences.append(vector_emotion.confidence)

        if llm_emotion and llm_emotion.confidence > 0:
            emotions.append(llm_emotion)
            confidences.append(llm_emotion.confidence)

        if heuristic_emotion and heuristic_emotion.confidence > 0:
            emotions.append(heuristic_emotion)
            confidences.append(heuristic_emotion.confidence)

        # Exit early if no valid emotions were generated
        if not emotions:
            return self._create_default_emotion(term)

        # Create a merged emotion vector with intelligent dimension integration
        merged_dimensions: Dict[EmotionDimension, float] = {}
        total_confidence = sum(confidences)

        # Normalize confidences to form weights
        weights = [float(conf) / total_confidence for conf in confidences]

        # Gather all dimensions from all vectors
        all_dimensions: Set[EmotionDimension] = set()
        for emotion in emotions:
            all_dimensions.update(emotion.dimensions.keys())

        # Calculate weighted values for each dimension
        for dimension in all_dimensions:
            # Weighted sum of dimension values from each source
            dimension_values: List[float] = []
            dimension_weights: List[float] = []

            for emotion, weight in zip(emotions, weights):
                if dimension in emotion.dimensions:
                    dimension_values.append(emotion.dimensions[dimension])
                    dimension_weights.append(weight)

            if dimension_weights:
                # Normalize dimension-specific weights
                dimension_weight_sum = sum(dimension_weights)
                normalized_weights = [float(w) / dimension_weight_sum for w in dimension_weights]

                # Calculate weighted dimension value
                merged_dimensions[dimension] = sum(
                    float(v) * float(w) for v, w in zip(dimension_values, normalized_weights)
                )

        # Calculate overall confidence as weighted average plus a synergy bonus
        # for having multiple sources agree
        base_confidence = sum(float(c) * float(w) for c, w in zip(confidences, weights))
        source_count_bonus = 0.05 * (len(emotions) - 1)  # Slight bonus for multiple sources
        merged_confidence = min(0.95, base_confidence + source_count_bonus)

        return EmotionVector(dimensions=merged_dimensions, confidence=merged_confidence)

    def _generate_vector_based_emotion(self, term: str) -> EmotionVector:
        """Generate emotion using word vectors and reference terms."""
        # Check if NLP is available
        if self.nlp is None:
            # Fallback to heuristic if NLP is not available
            return self._generate_heuristic_emotion(term)

        # Process the term with spaCy
        doc = self.nlp(term)
        # Reference emotion words for comparison
        reference_emotions = {
            "happy": (EmotionDimension.VALENCE, 0.8),
            "sad": (EmotionDimension.VALENCE, -0.8),
            "angry": (EmotionDimension.AROUSAL, 0.9),
            "calm": (EmotionDimension.AROUSAL, -0.7),
            "powerful": (EmotionDimension.DOMINANCE, 0.9),
            "weak": (EmotionDimension.DOMINANCE, -0.8),
            "certain": (EmotionDimension.CERTAINTY, 0.9),
            "uncertain": (EmotionDimension.CERTAINTY, -0.8),
            "important": (EmotionDimension.RELEVANCE, 0.9),
            "trivial": (EmotionDimension.RELEVANCE, -0.7),
            "novel": (EmotionDimension.NOVELTY, 0.9),
            "familiar": (EmotionDimension.NOVELTY, -0.7),
            "responsible": (EmotionDimension.AGENCY, 0.8),
            "helpless": (EmotionDimension.AGENCY, -0.8),
            "connected": (EmotionDimension.SOCIAL, 0.9),
            "isolated": (EmotionDimension.SOCIAL, -0.9),
            "future": (EmotionDimension.TEMPORAL, 0.7),
            "past": (EmotionDimension.TEMPORAL, -0.7),
        }

        # Update type hint for reference_docs to use proper spaCy Doc type
        try:
            from spacy.tokens import Doc

            self._reference_docs: Dict[str, Doc] = {}
        except ImportError:
            self._reference_docs = {}

        # Pre-process reference words once
        if not self._reference_docs:
            for ref_word in reference_emotions:
                self._reference_docs[ref_word] = self.nlp(ref_word)
                self._reference_docs[ref_word] = self.nlp(ref_word)

        # Calculate emotional dimensions based on similarity to reference words
        dimensions: Dict[EmotionDimension, float] = {}
        for ref_word, (dimension, ref_value) in reference_emotions.items():
            # Access the spaCy Doc object for the reference word
            ref_doc = self._reference_docs.get(ref_word)
            if not ref_doc:
                continue

            # Calculate similarity
            similarity = max(0, min(1, doc.similarity(ref_doc)))

            # Scale the dimension value based on similarity and reference value
            if similarity > 0.3:  # Only consider meaningful similarities
                if dimension not in dimensions:
                    dimensions[dimension] = 0

                # Add weighted contribution
                if ref_value > 0:
                    dimensions[dimension] += similarity * ref_value
                else:
                    dimensions[dimension] -= similarity * abs(ref_value)

        # Normalize values using the helper method
        dimensions = self._normalize_dimensions(dimensions)

        # Ensure primary dimensions are present
        for dim in EmotionDimension.primary_dimensions():
            if dim not in dimensions:
                dimensions[dim] = 0.0

        return EmotionVector(dimensions=dimensions, confidence=0.75)

    def _generate_llm_based_emotion(self, term: str) -> Optional[EmotionVector]:
        """
        Generate emotion vector using LLM for semantic understanding.

        Leverages the language model to analyze emotional dimensions
        of the term with advanced linguistic understanding.

        Args:
            term: Term to analyze emotionally

        Returns:
            Emotion vector or None if LLM analysis fails
        """
        try:
            # Verify LLM availability in emotion manager
            if not hasattr(self.emotion_manager, "llm_interface") or self.emotion_manager.llm_interface is None:
                return None

            # Construct a structured prompt for emotional analysis
            prompt = f"""Analyze the emotional connotations of the word '{term}'.

    Provide numerical values in this format:
    {{
      "valence": [value between -1.0 and 1.0, negative is unpleasant, positive is pleasant],
      "arousal": [value between 0.0 and 1.0, where 0 is calm, 1 is excited],
      "dominance": [value between 0.0 and 1.0, where 0 is submissive, 1 is dominant]
    }}"""

            # Generate analysis with controlled parameters
            response = self.emotion_manager.llm_interface.generate_text(
                prompt,
                max_new_tokens=100,
                temperature=0.1,  # Low temperature for consistent results
            )

            if not response:
                return None

            # Extract JSON from response
            json_match = re.search(r"{.*}", response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    # Extract and validate values
                    valence = max(-1.0, min(1.0, float(data.get("valence", 0))))
                    arousal = max(0.0, min(1.0, float(data.get("arousal", 0.5))))
                    dominance = max(0.0, min(1.0, float(data.get("dominance", 0.5))))

                    # Construct emotion vector
                    dimensions = {
                        EmotionDimension.VALENCE: valence,
                        EmotionDimension.AROUSAL: arousal,
                        EmotionDimension.DOMINANCE: dominance,
                    }
                    return EmotionVector(dimensions=dimensions, confidence=0.85)
                except (json.JSONDecodeError, ValueError):
                    pass

            # Fallback to regex parsing if JSON extraction fails
            valence_match = re.search(r"valence[:\s]+([-+]?\d*\.?\d+)", response, re.IGNORECASE)
            arousal_match = re.search(r"arousal[:\s]+(\d*\.?\d+)", response, re.IGNORECASE)

            if valence_match and arousal_match:
                valence = max(-1.0, min(1.0, float(valence_match.group(1))))
                arousal = max(0.0, min(1.0, float(arousal_match.group(1))))

                dimensions = {
                    EmotionDimension.VALENCE: valence,
                    EmotionDimension.AROUSAL: arousal,
                }
                return EmotionVector(dimensions=dimensions, confidence=0.7)

            return None
        except Exception as e:
            logger.debug(f"LLM-based emotion generation failed for '{term}': {str(e)}")
            return None

    def _generate_heuristic_emotion(self, term: str) -> EmotionVector:
        """Generate emotion vector using text patterns and heuristics."""
        # Get consistent hash-based values for the term
        hash_val = int(hashlib.md5(term.encode()).hexdigest(), 16)

        # Basic valence: positive/negative sentiment heuristic
        positive_patterns = r"happ|joy|love|good|great|nice|win|pleasant|delight|content"
        negative_patterns = r"sad|anger|hate|bad|awful|terr|fear|anxi|depress|rage|grief"

        valence = 0.0
        if re.search(positive_patterns, term.lower()):
            valence = 0.5
        elif re.search(negative_patterns, term.lower()):
            valence = -0.5

        # Add randomness from hash for uniqueness
        valence += ((hash_val % 100) / 100 - 0.5) * 0.5

        # Basic arousal: word length, consonant density, etc.
        consonant_count = len(re.findall(r"[bcdfghjklmnpqrstvwxz]", term.lower()))
        consonant_ratio = consonant_count / max(1, len(term))

        # Arousal patterns
        high_arousal = r"excit|thrill|energe|passion|intens|ecsta|vigor|alert|awake"
        low_arousal = r"calm|relax|tranquil|peace|serene|gentle|quiet|sleep"

        arousal_base = 0.0
        if re.search(high_arousal, term.lower()):
            arousal_base = 0.5
        elif re.search(low_arousal, term.lower()):
            arousal_base = -0.5

        arousal = arousal_base + (consonant_ratio - 0.5) + ((hash_val // 100) % 100) / 100 - 0.5

        # Basic dominance
        high_dominance = r"power|control|domin|master|confiden|assert|strong"
        low_dominance = r"weak|submiss|vulnerab|helpless|fragile|timid"

        dominance_base = 0.0
        if re.search(high_dominance, term.lower()):
            dominance_base = 0.5
        elif re.search(low_dominance, term.lower()):
            dominance_base = -0.5

        dominance = dominance_base + ((hash_val // 10000) % 100) / 100 - 0.5

        # Create dimensions dictionary with normalized values
        dimensions = {
            EmotionDimension.VALENCE: valence,
            EmotionDimension.AROUSAL: arousal,
            EmotionDimension.DOMINANCE: dominance,
        }

        dimensions = self._normalize_dimensions(dimensions)

        # Return with lower confidence as this is generated
        return EmotionVector(dimensions=dimensions, confidence=0.7)

    def _create_default_emotion(self, term: str = "") -> EmotionVector:
        """Create a neutral emotion vector with low confidence."""
        # For empty terms, return neutral vector with low confidence
        if not term:
            return EmotionVector(
                dimensions={
                    EmotionDimension.VALENCE: 0.0,
                    EmotionDimension.AROUSAL: 0.0,
                    EmotionDimension.DOMINANCE: 0.0,
                },
                confidence=0.5,
            )

        # For actual terms, attempt to extract some information
        # with even lower confidence
        return self._generate_heuristic_emotion(term)

    def _create_fallback_concept(self, term: str) -> EmotionalConcept:
        """Create a simplified emotional concept when recursion limit is reached."""
        return EmotionalConcept(
            term=term,
            word_id=-1,  # Placeholder ID
            primary_emotion=self._create_default_emotion(term),
        )

    def _add_secondary_emotions(self, concept: EmotionalConcept) -> None:
        """Add secondary emotions based on relationships."""
        # Get related words for this concept
        related_terms = self._get_related_terms(concept.word_id)

        for rel_type, rel_term, rel_strength in related_terms:
            # Skip if we've reached recursion limit
            if self._processing_depth >= self._max_recursion:
                # Record relationship but don't process further
                concept.add_related_context(rel_type, rel_term, rel_strength)
                continue

            # Process only certain relationship types for secondary emotions
            if rel_type in (
                "synonym",
                "emotional_synonym",
                "related",
                "emotional_component",
                "intensifies",
                "diminishes",
                "evokes",
                "hypernym",
                "hyponym",
            ):
                # Get emotion for related term
                rel_emotion = self.get_emotion_vector(rel_term)

                # Add as secondary emotion with appropriate weighting
                weighted_emotion = rel_emotion
                if rel_type == "intensifies":
                    weighted_emotion = concept.primary_emotion.intensify()
                elif rel_type == "diminishes":
                    weighted_emotion = concept.primary_emotion.diminish()

                concept.secondary_emotions.append((f"{rel_type}:{rel_term}", weighted_emotion))

                # Record the relationship context
                concept.add_related_context(rel_type, rel_term, rel_strength)

    def _add_meta_emotions(self, concept: EmotionalConcept) -> None:
        """Add meta-emotions (emotions about emotions)."""
        # Skip if we've reached recursion limit
        if self._processing_depth >= self._max_recursion:
            return

        # Create meta-emotions based on primary emotion characteristics
        primary = concept.primary_emotion

        # Meta-emotion: Emotional clarity/awareness
        valence = primary.dimensions.get(EmotionDimension.VALENCE, 0.0)
        if abs(valence) > 0.7:  # Strong valence
            # Meta-emotion about having a strong emotion
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

        # Meta-emotion: Response to arousal
        arousal = primary.dimensions.get(EmotionDimension.AROUSAL, 0.0)
        if arousal > 0.7:  # High arousal
            # Meta-emotion about high activation
            meta_dimensions = {
                EmotionDimension.VALENCE: (-0.2 if arousal > 0.9 else 0.2),  # Negative if extreme
                EmotionDimension.AROUSAL: -0.5,  # Low arousal response to high arousal
                EmotionDimension.DOMINANCE: (0.3 if arousal < 0.9 else -0.3),  # Control except at extremes
                EmotionDimension.META_STABILITY: -0.4,  # Recognition of instability
            }
            concept.add_meta_emotion(
                "response_to_activation",
                EmotionVector(dimensions=meta_dimensions, confidence=0.7),
            )

        # Meta-emotion: Emotional complexity
        dimensions_count = len([v for v in primary.dimensions.values() if abs(v) > 0.5])
        if dimensions_count >= 3:  # Complex emotion with multiple strong dimensions
            meta_dimensions = {
                EmotionDimension.VALENCE: 0.2,  # Slightly positive about complexity
                EmotionDimension.AROUSAL: 0.1,  # Slight arousal increase
                EmotionDimension.CERTAINTY: -0.3,  # Lower certainty with complexity
                EmotionDimension.META_COMPLEXITY: 0.9,  # High complexity awareness
            }
            concept.add_meta_emotion(
                "awareness_of_emotional_complexity",
                EmotionVector(dimensions=meta_dimensions, confidence=0.7),
            )

        # Meta-emotion: Emotional congruence
        # (for terms that have secondary emotions)
        if concept.secondary_emotions:
            # Check if secondary emotions align with or contradict primary
            congruent = True
            for _, second_emotion in concept.secondary_emotions:
                val1 = primary.dimensions.get(EmotionDimension.VALENCE, 0.0)
                val2 = second_emotion.dimensions.get(EmotionDimension.VALENCE, 0.0)
                if val1 * val2 < 0 and abs(val1) > 0.3 and abs(val2) > 0.3:
                    congruent = False
                    break

            if not congruent:
                meta_dimensions = {
                    EmotionDimension.VALENCE: -0.3,  # Negative response to incongruence
                    EmotionDimension.AROUSAL: 0.4,  # Increased arousal from conflict
                    EmotionDimension.CERTAINTY: -0.5,  # Lower certainty
                    EmotionDimension.META_CONGRUENCE: -0.8,  # Very low congruence
                }
                concept.add_meta_emotion(
                    "response_to_emotional_incongruence",
                    EmotionVector(dimensions=meta_dimensions, confidence=0.6),
                )

    def _add_emotional_patterns(self, concept: EmotionalConcept) -> None:
        """Add emotional patterns like sequences or context-specific variations."""
        # Skip if we've reached recursion limit
        if self._processing_depth >= self._max_recursion:
            return

        # Create a basic emotional arc (sequence over time)
        primary = concept.primary_emotion

        # Pattern: emotional timeline (rise and fall)
        self._add_temporal_sequence(concept, primary)

        # Pattern: intensity gradation
        self._add_intensity_gradation(concept, primary)

        # Pattern: contextual variations
        self._add_contextual_variations(concept, primary)

    def _add_temporal_sequence(self, concept: EmotionalConcept, primary: EmotionVector) -> None:
        """Add temporal sequence pattern (how the emotion evolves over time)."""
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
            end_dims[EmotionDimension.VALENCE] *= 0.8
        timeline.append(EmotionVector(dimensions=end_dims, confidence=end_emotion.confidence))

        concept.add_emotional_pattern("temporal_sequence", timeline)

    def _add_intensity_gradation(self, concept: EmotionalConcept, primary: EmotionVector) -> None:
        """Add intensity gradation pattern (emotion at different intensity levels)."""
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
            dimensions={dim: max(-1.0, min(1.0, val * 1.5)) for dim, val in primary.dimensions.items()},
            confidence=0.8,
        )
        gradation.append(high)

        # Extreme intensity (may have different qualities)
        extreme_dims = {dim: max(-1.0, min(1.0, val * 2.0)) for dim, val in primary.dimensions.items()}

        # Extreme intensity often has different emotional qualities
        # Add some complexity/instability for extreme emotions
        if EmotionDimension.CERTAINTY not in extreme_dims:
            extreme_dims[EmotionDimension.CERTAINTY] = -0.3
        if EmotionDimension.SOCIAL not in extreme_dims:
            extreme_dims[EmotionDimension.SOCIAL] = -0.2

        extreme = EmotionVector(dimensions=extreme_dims, confidence=0.7)
        gradation.append(extreme)

        concept.add_emotional_pattern("intensity_gradation", gradation)

    def _add_contextual_variations(self, concept: EmotionalConcept, primary: EmotionVector) -> None:
        """Add contextual variations pattern (emotion in different contexts)."""
        variations: List[EmotionVector] = []

        # Personal/intimate context
        personal_dims = dict(primary.dimensions)
        personal_dims[EmotionDimension.INTENSITY] = min(1.0, personal_dims.get(EmotionDimension.INTENSITY, 0) + 0.3)
        personal_dims[EmotionDimension.RELEVANCE] = min(1.0, personal_dims.get(EmotionDimension.RELEVANCE, 0) + 0.4)
        personal_dims[EmotionDimension.SOCIAL] = min(1.0, personal_dims.get(EmotionDimension.SOCIAL, 0) + 0.5)
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
        if EmotionDimension.VALENCE in social_dims and social_dims[EmotionDimension.VALENCE] < 0:
            # Negative emotions often muted in social contexts
            social_dims[EmotionDimension.VALENCE] = social_dims[EmotionDimension.VALENCE] * 0.5
        social_dims[EmotionDimension.SOCIAL] = min(1.0, social_dims.get(EmotionDimension.SOCIAL, 0) + 0.6)
        variations.append(EmotionVector(dimensions=social_dims, confidence=0.7))

        # Cultural context
        cultural_dims = dict(primary.dimensions)
        # Cultural contexts often modify emotional display rules
        cultural_dims[EmotionDimension.INTENSITY] = max(-1.0, cultural_dims.get(EmotionDimension.INTENSITY, 0) - 0.2)
        cultural_dims[EmotionDimension.CERTAINTY] = max(-1.0, cultural_dims.get(EmotionDimension.CERTAINTY, 0) - 0.3)
        variations.append(EmotionVector(dimensions=cultural_dims, confidence=0.6))

        concept.add_emotional_pattern("contextual_variations", variations)

    def _get_related_terms(self, word_id: int) -> List[Tuple[str, str, float]]:
        """Get related terms and relationship types for a word with proper fallbacks."""
        try:
            # Primary query with relationship_types join for weights
            primary_query = """
            SELECT r.relationship_type, w2.term,
                   COALESCE(rt.weight, 0.5) as weight
            FROM relationships r
            JOIN words w1 ON r.word_id = w1.id
            JOIN words w2 ON r.related_term = w2.term
            LEFT JOIN relationship_types rt ON r.relationship_type = rt.type
            WHERE w1.id = ?
            LIMIT 15
            """

            try:
                with self.db.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(primary_query, (word_id,))
                    results = [(row[0], row[1], float(row[2])) for row in cursor.fetchall()]
                    if results:
                        return results
            except sqlite3.OperationalError:
                # Fall through to fallback queries
                pass

            # First fallback - query without relationship_types join
            fallback_query = """
            SELECT r.relationship_type, w2.term, 0.5 as weight
            FROM relationships r
            JOIN words w1 ON r.word_id = w1.id
            JOIN words w2 ON r.related_term = w2.term
            WHERE w1.id = ?
            LIMIT 15
            """

            try:
                with self.db.create_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(fallback_query, (word_id,))
                    return [(row[0], row[1], float(row[2])) for row in cursor.fetchall()]
            except sqlite3.Error as e:
                print(f"Database error fetching relationships: {e}")

        except Exception as e:
            print(f"Error getting related terms: {e}")

        # Ultimate fallback uses wordnet and other heuristics
        return self._get_heuristic_related_terms(word_id)

    def _get_heuristic_related_terms(self, word_id: int) -> List[Tuple[str, str, float]]:
        """Fallback method to generate plausible related terms using NLP."""
        # Get the term for this word_id
        term = None
        try:
            query = "SELECT term FROM words WHERE id = ?"
            with self.db.create_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (word_id,))
                row = cursor.fetchone()
                if row:
                    term = row[0]
        except Exception:
            return []

        if not term or not self.nlp:
            return []

        # Use spaCy to find similar words
        term_vector = self._get_vector(term)
        if term_vector is None:
            return []

        # Find similar words in vocabulary
        result: List[Tuple[str, str, float]] = []
        seen: set[str] = set()

        # Check if we have a cached result
        cache_key = f"similar_to_{term}"
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]

        # Find synonyms using vector similarity
        similar_terms: List[Tuple[str, float]] = []
        # Process the term with spaCy to get a document object
        doc = self.nlp(term)
        if doc.has_vector:
            # Get words from vocabulary that have vectors
            for word in self.nlp.vocab:
                # Skip words without vectors, non-alphabetic words, and the term itself
                if word.has_vector and word.is_alpha and word.text.lower() != term.lower() and len(word.text) > 2:
                    similarity = word.similarity(doc)
                    if similarity > 0.6:  # Only consider reasonably similar words
                        similar_terms.append((word.text, similarity))

            # Sort by similarity and take top 5
            similar_terms.sort(key=lambda x: x[1], reverse=True)
            for similar_term, similarity in similar_terms[:5]:
                if similar_term not in seen:
                    result.append(("synonym", similar_term, float(similarity)))
                    seen.add(similar_term)

        # Use additional heuristics based on term properties
        # For example, prefix/suffix modifications suggest related words
        if len(term) > 5:
            # Potential derived forms
            suffixes = ["ing", "ed", "er", "ion", "ity", "ment", "ness", "ly"]
            prefixes = ["un", "re", "in", "dis", "over", "under"]

            # Check potential derivations
            for suffix in suffixes:
                if term.endswith(suffix):
                    base = term[: -len(suffix)]
                    if len(base) > 3:
                        if base not in seen:
                            result.append(("derived_from", base, 0.7))
                            seen.add(base)
                else:
                    derived = term + suffix
                    if derived not in seen:
                        result.append(("hyponym", derived, 0.6))
                        seen.add(derived)

            for prefix in prefixes:
                if term.startswith(prefix):
                    base = term[len(prefix) :]
                    if len(base) > 3:
                        if base not in seen:
                            result.append(("derived_from", base, 0.7))
                            seen.add(base)
                else:
                    derived = prefix + term
                    if derived not in seen:
                        if prefix in ("un", "in", "dis"):
                            result.append(("antonym", derived, 0.6))
                        else:
                            result.append(("related", derived, 0.6))
                        seen.add(derived)

        return result

    def _calculate_evocative_strength(self, emotion1: EmotionVector, emotion2: EmotionVector) -> float:
        """Calculate how strongly one emotion might evoke another."""
        # Check if dimensions correlate in expected ways for evocation
        valence1 = emotion1.dimensions.get(EmotionDimension.VALENCE, 0.0)
        valence2 = emotion2.dimensions.get(EmotionDimension.VALENCE, 0.0)
        arousal1 = emotion1.dimensions.get(EmotionDimension.AROUSAL, 0.0)
        arousal2 = emotion2.dimensions.get(EmotionDimension.AROUSAL, 0.0)
        dominance1 = emotion1.dimensions.get(EmotionDimension.DOMINANCE, 0.0)

        # Common evocation patterns:
        pattern_strengths: List[float] = []

        # 1. Negative valence often evokes high arousal
        if valence1 < -0.3 and arousal2 > 0.3:
            pattern_strengths.append(min(abs(valence1), arousal2))

        # 2. Extreme valence (either way) tends to evoke matching valence
        if abs(valence1) > 0.7 and abs(valence2) > 0.5:
            if (valence1 > 0 and valence2 > 0) or (valence1 < 0 and valence2 < 0):
                pattern_strengths.append(min(abs(valence1), abs(valence2)) * 0.7)

        # 3. High arousal with low dominance can evoke anxiety-like responses
        if arousal1 > 0.5 and dominance1 < -0.3 and arousal2 > 0.5 and valence2 < -0.3:
            pattern_strengths.append(min(arousal1, abs(dominance1)) * 0.8)

        # 4. Low arousal with high dominance can evoke contentment
        if arousal1 < -0.3 and dominance1 > 0.5 and valence2 > 0.5 and arousal2 < 0:
            pattern_strengths.append(min(abs(arousal1), dominance1) * 0.7)

        return max(pattern_strengths + [0.2])  # Base connection as fallback

    def _calculate_component_strength(self, component: EmotionVector, composite: EmotionVector) -> float:
        """Calculate how likely component is a part of composite emotion."""
        # Components have fewer strong dimensions than composites
        component_dims = sum(1 for v in component.dimensions.values() if abs(v) > 0.5)
        composite_dims = sum(1 for v in composite.dimensions.values() if abs(v) > 0.5)

        if component_dims >= composite_dims:
            return 0.2  # Components should be simpler

        # Check if component dimensions are present in composite with similar values
        matching_dims = 0
        for dim, val in component.dimensions.items():
            if abs(val) > 0.4 and dim in composite.dimensions:
                comp_val = composite.dimensions[dim]
                if abs(comp_val) > 0.4 and (val * comp_val > 0):  # Same direction
                    matching_dims += 1

        # Calculate strength
        if matching_dims == 0:
            return 0.2  # No significant overlap

        # Component should be largely contained within composite
        return min(1.0, matching_dims / max(1, component_dims) * 0.8)

    def _normalize_dimensions(self, dimensions: Dict[EmotionDimension, float]) -> Dict[EmotionDimension, float]:
        """Normalize dimension values to ensure they remain in valid range.

        Args:
            dimensions: Dictionary mapping emotional dimensions to values

        Returns:
            Dictionary with all values normalized to [-1.0, 1.0] range
        """
        return {dim: max(-1.0, min(1.0, val)) for dim, val in dimensions.items()}

    def _get_cache_key(self, term: str, context: Optional[EmotionalContext] = None) -> str:
        """Generate a cache key for a term and optional context.

        Args:
            term: The word or phrase to process
            context: Optional emotional context to consider

        Returns:
            String key that uniquely identifies the term+context combination
        """
        if not context:
            return term

        # Create a hash of the context's dimension adjustments
        ctx_factors: List[Tuple[str, float]] = []
        if context.domain_specific:
            ctx_factors.extend([(str(k), float(v)) for k, v in sorted(context.domain_specific.items())])
        if context.cultural_factors:
            ctx_factors.extend([(str(k), float(v)) for k, v in sorted(context.cultural_factors.items())])
        if context.situational_factors:
            ctx_factors.extend([(str(k), float(v)) for k, v in sorted(context.situational_factors.items())])
        if context.temporal_factors:
            ctx_factors.extend([(str(k), float(v)) for k, v in sorted(context.temporal_factors.items())])

        # Generate a deterministic hash of context factors
        ctx_hash = hashlib.md5(str(ctx_factors).encode()).hexdigest()[:8]
        return f"{term}::{ctx_hash}"

    def _get_vector(self, term: str) -> Optional[Any]:
        """Get cached or fresh word vector.

        Args:
            term: Word to get vector for

        Returns:
            Word vector or None if not available
        """
        if term in self._vector_cache:
            return self._vector_cache[term]

        if not self.nlp:
            return None

        try:
            doc = self.nlp(term)
            if doc.has_vector:
                self._vector_cache[term] = doc.vector
                return doc.vector
        except Exception as e:
            print(f"Error getting vector for '{term}': {e}")

        return None

    @eidosian()
    def register_meta_emotion_hook(self, hook: Callable[[EmotionalConcept], None]) -> None:
        """Register a new meta-emotion generation hook.

        Args:
            hook: Function that adds meta-emotions to a concept
        """
        self.meta_emotion_hooks.append(hook)

    @eidosian()
    def register_pattern_hook(self, hook: Callable[[EmotionalConcept], None]) -> None:
        """Register a new emotional pattern generation hook.

        Args:
            hook: Function that adds emotional patterns to a concept
        """
        self.pattern_hooks.append(hook)

    def _register_default_hooks(self) -> None:
        """Register the built-in emotion processing hooks."""
        # Register meta-emotion hooks
        self.register_meta_emotion_hook(add_clarity_meta_emotion)
        self.register_meta_emotion_hook(add_arousal_meta_emotion)
        self.register_meta_emotion_hook(add_complexity_meta_emotion)
        self.register_meta_emotion_hook(add_congruence_meta_emotion)

        # Register pattern hooks
        self.register_pattern_hook(add_temporal_sequence)
        self.register_pattern_hook(add_intensity_gradation)
        self.register_pattern_hook(add_contextual_variations)

        # Special case: secondary emotions hook needs access to related terms
        # We'll need to create a closure that provides this functionality
        @eidosian()
        def secondary_emotions_with_db(concept: EmotionalConcept) -> None:
            related_terms = self._get_related_terms(concept.word_id)
            # Process related terms similar to _add_secondary_emotions
            for rel_type, rel_term, rel_strength in related_terms:
                if rel_strength > 0.5:  # Only use strong relationships
                    related_concept = self.process_term(rel_term)
                    if related_concept:
                        concept.add_secondary_emotion(f"{rel_type}_{rel_term}", related_concept.primary_emotion)

        self.register_pattern_hook(secondary_emotions_with_db)
