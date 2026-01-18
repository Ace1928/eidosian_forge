"""Emotional analysis and persistence system for text and lexical entities.

Provides dimensional, categorical, and contextual emotional processing through
a hybrid analysis architecture that combines:

- Dimensional analysis: Valence (-1.0 to 1.0) and arousal (0.0 to 1.0)
- Categorical classification: Discrete emotional states with confidence scores
- Multi-method sentiment fusion: TextBlob, VADER, and optional LLM integration
- Recursive emotional processing: Meta-emotions and emotional relationships
- Contextual analysis: Domain, cultural, and situational emotional adjustments
- Persistent storage: Database-agnostic emotion data with efficient retrieval

Core capabilities:
1. Text sentiment measurement with configurable analyzer weights
2. Message-level emotion classification and confidence scoring
3. Word-level emotion association tracking
4. Recursive emotional relationship analysis
5. Custom emotional context creation and management
6. Runtime optimization through performance metrics
7. Dialect-independent database operations

This system progressively enhances emotional understanding based on available
analyzers, providing graceful degradation when advanced components like
VADER or LLM are unavailable.

The EmotionManager serves as the unified interface to all emotional processing
operations while maintaining backward compatibility with earlier emotion
analysis implementations.
"""

import json

# Initialize logger for the class
import logging
import sqlite3
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast

try:  # Optional dependencies
    from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore

    VADER_AVAILABLE = True
except Exception:  # pragma: no cover - allow missing VADER
    SentimentIntensityAnalyzer = None  # type: ignore
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob  # type: ignore
except Exception:  # pragma: no cover - allow missing TextBlob
    TextBlob = None  # type: ignore

from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_config import (
    EmotionCategory,
    EmotionCategoryType,
    EmotionConfig,
    EmotionDetectionMetrics,
    normalize_emotion_category,
)
from word_forge.emotion.emotion_types import (
    EmotionalContext,
    EmotionAnalysisDict,
    EmotionError,
    FullEmotionAnalysisDict,
    MessageEmotionDict,
    VaderSentimentScores,
    WordEmotionDict,
)

# LLM Interface
from word_forge.parser.language_model import ModelState

logger = logging.getLogger(__name__)
LLM_AVAILABLE = True

# SQL statement for storing emotional relationships derived from word analysis
SQL_UPSERT_EMOTIONAL_RELATIONSHIP = """
INSERT INTO emotional_relationships
    (word_id, related_term, relationship_type, valence, arousal, last_updated)
VALUES
    (?, ?, ?, ?, ?, ?)
ON CONFLICT(word_id, related_term, relationship_type)
DO UPDATE SET
    valence=excluded.valence,
    arousal=excluded.arousal,
    last_updated=excluded.last_updated
"""

# Set of positive emotion categories for efficient categorization
POSITIVE_EMOTIONS = {EmotionCategory.HAPPINESS, EmotionCategory.SURPRISE}
# Set of negative emotion categories for efficient categorization
NEGATIVE_EMOTIONS = {
    EmotionCategory.SADNESS,
    EmotionCategory.ANGER,
    EmotionCategory.FEAR,
    EmotionCategory.DISGUST,
}


class EmotionManager:
    """Tracks emotional associations with words and conversation messages.

    This class manages the storage and retrieval of emotional data:
    - Word emotions: valence (positive/negative) and arousal levels
    - Message emotions: categorical emotion labels with confidence scores

    All emotional data is persistent and stored in database tables.

    The emotion analysis uses a hybrid approach combining TextBlob, VADER
    (when available), and optional LLM integration for more accurate and
    nuanced sentiment detection in varied texts.

    Advanced recursive emotion processing enables deeper analysis of
    emotional relationships, meta-emotions, and emotional patterns.

    Attributes:
        db_manager: Database manager providing connection to storage
        config: Configuration for emotion analysis parameters
        metrics: Detection metrics for runtime performance optimization
        VALENCE_RANGE: Valid range for valence values (-1.0 to 1.0 by default)
        AROUSAL_RANGE: Valid range for arousal values (0.0 to 1.0 by default)
        CONFIDENCE_RANGE: Valid range for confidence values (0.0 to 1.0 by default)
        vader_weight: Weight given to VADER in hybrid analysis
        textblob_weight: Weight given to TextBlob in hybrid analysis
        llm_weight: Weight given to LLM analysis when available
        vader: VADER sentiment analyzer instance (if available)
        llm_interface: LLM interface for enhanced emotional analysis (if available)
        _recursive_processor: Cached instance of recursive emotion processor
    """

    def __init__(self, db_manager: DBManager) -> None:
        """Initialize the emotion tracking system.

        Establishes database connection, creates necessary tables, and
        initializes sentiment analysis tools. Sets up all configuration
        parameters to ensure consistent emotional measurement across
        the system.

        Args:
            db_manager: Database manager providing connection to storage

        Raises:
            EmotionError: If there's an issue initializing the emotion tables
                or connecting to required resources
        """
        self.db_manager = db_manager

        # Create config instance
        self.config = EmotionConfig()

        # Add metrics for runtime optimization
        self.metrics = EmotionDetectionMetrics()

        self._create_tables()

        # Initialize sentiment analyzers and LLM if available
        self._init_analysis_tools()

        # Access emotion constraints from centralized config instance
        self.VALENCE_RANGE = self.config.valence_range
        self.AROUSAL_RANGE = self.config.arousal_range
        self.CONFIDENCE_RANGE = self.config.confidence_range
        self.EMOTION_KEYWORDS = self.config.emotion_keywords

        # Hybrid analysis weights
        self.vader_weight = self.config.vader_weight
        self.textblob_weight = self.config.textblob_weight
        self.llm_weight = getattr(self.config, "llm_weight", 0.0)

        # Initialize recursive emotion processor lazily
        self._recursive_processor = None

        # Track detection counts for optimization
        self._detection_count = 0
        self._optimization_frequency = 100  # Optimize after this many detections

    @property
    def recursive_processor(self):
        """Lazy initialization of recursive emotion processor.

        Returns:
            RecursiveEmotionProcessor: Emotion processor for recursive analysis
        """
        if self._recursive_processor is None:
            # Import here to avoid circular import issues
            from word_forge.emotion.emotion_processor import RecursiveEmotionProcessor

            self._recursive_processor = RecursiveEmotionProcessor(
                db_manager=self.db_manager, emotion_manager=self
            )
        return self._recursive_processor

    def _init_analysis_tools(self) -> None:
        """Initialize sentiment analysis tools based on availability."""
        # Initialize VADER if available
        try:
            if VADER_AVAILABLE:
                self.vader = SentimentIntensityAnalyzer()
            else:
                self.vader = None
        except Exception as e:
            self.vader = None
            logger.warning(f"VADER initialization failed: {str(e)}")

        # Initialize LLM if available
        try:
            if LLM_AVAILABLE:
                self.llm_interface = ModelState()
                if self.llm_interface.initialize():
                    logger.info(
                        f"LLM initialized successfully: {self.llm_interface.model_name}"
                    )
                else:
                    self.llm_interface = None
                    logger.warning(
                        "LLM initialization failed - operating with reduced capabilities"
                    )
            else:
                self.llm_interface = None
        except Exception as e:
            self.llm_interface = None
            logger.warning(f"LLM initialization failed: {str(e)}")

    def init_analysis_tools(self) -> None:
        """Reinitialize analysis tools.

        This method allows for reinitialization of the sentiment analysis
        tools, useful in cases where the configuration has changed or
        the system needs to refresh its resources.
        """
        self._init_analysis_tools()

    @contextmanager
    def _db_connection(self):
        """Create a database connection using the DBManager's path.

        Creates and manages a database connection with SQLite's Row factory
        for dictionary-like access to results. Automatically closes the
        connection when exiting the context.

        Yields:
            sqlite3.Connection: An active database connection

        Example:
            ```python
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM word_emotion")
                results = cursor.fetchall()
            # Connection automatically closed after block
            ```
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _create_tables(self) -> None:
        """Initialize database tables for emotion storage.

        Creates two tables if they don't exist:
        - word_emotion: Links words to valence and arousal scores
        - message_emotion: Links messages to categorical emotion labels

        Table schemas are defined in the EmotionConfig class to ensure
        consistency across system components. Uses dialect-specific SQL
        based on the configuration.

        Raises:
            EmotionError: If there's an issue creating the emotion tables
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Use dialect-specific SQL templates
                cursor.execute(
                    self.config.get_sql_template("create_word_emotion_table")
                )
                cursor.execute(
                    self.config.get_sql_template("create_message_emotion_table")
                )
                conn.commit()
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to initialize emotion tables: {e}",
                {"operation": "create_tables", "db_path": self.db_manager.db_path},
            ) from e

    @lru_cache(maxsize=256)
    def _clamp_emotional_values(
        self, valence: float, arousal: float
    ) -> Tuple[float, float]:
        """Clamp valence and arousal values to their valid ranges."""
        valence = max(self.VALENCE_RANGE[0], min(self.VALENCE_RANGE[1], valence))
        arousal = max(self.AROUSAL_RANGE[0], min(self.AROUSAL_RANGE[1], arousal))
        return valence, arousal

    def set_word_emotion(self, word_id: int, valence: float, arousal: float) -> None:
        """Store or update emotional values for a word.

        Assigns valence (positivity/negativity) and arousal (intensity)
        values to a word, clamping values to valid ranges defined in
        configuration.

        Args:
            word_id: Database ID of the target word
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal/intensity (0.0 to 1.0)

        Raises:
            EmotionError: If the database operation fails

        Example:
            ```python
            # Set "happy" as strongly positive with moderate arousal
            word_id = db_manager.get_word_id("happy")
            emotion_manager.set_word_emotion(word_id, 0.8, 0.6)
            ```
        """
        valence, arousal = self._clamp_emotional_values(valence, arousal)

        relationship_type, anchor_term = self._derive_emotional_relationship(
            valence, arousal
        )
        self._ensure_emotion_anchor(anchor_term)

        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                timestamp = time.time()
                cursor.execute(
                    self.config.get_sql_template("insert_word_emotion"),
                    (word_id, valence, arousal, timestamp),
                )
                self._store_emotional_relationship(
                    cursor,
                    word_id,
                    anchor_term,
                    relationship_type,
                    valence,
                    arousal,
                    timestamp,
                )
                conn.commit()
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to store word emotion: {e}",
                {"word_id": word_id, "valence": valence, "arousal": arousal},
            ) from e

    @lru_cache(maxsize=256)
    def get_word_emotion(self, word_id: int) -> Optional[WordEmotionDict]:
        """Retrieve emotional data for a word.

        Args:
            word_id: Database ID of the target word

        Returns:
            Dictionary containing emotional data (valence, arousal, timestamp),
            or None if no data exists for the word

        Raises:
            EmotionError: If the database operation fails

        Example:
            ```python
            word_id = db_manager.get_word_id("happy")
            emotion_data = emotion_manager.get_word_emotion(word_id)
            if emotion_data:
                print(f"Valence: {emotion_data['valence']}")
                print(f"Arousal: {emotion_data['arousal']}")
            ```
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self.config.get_sql_template("get_word_emotion"), (word_id,)
                )
                row: Optional[sqlite3.Row] = cursor.fetchone()

                if row:
                    return WordEmotionDict(
                        word_id=int(row["word_id"]),
                        valence=float(row["valence"]),
                        arousal=float(row["arousal"]),
                        timestamp=float(row["timestamp"]),
                        dominance=(
                            float(row["dominance"])
                            if "dominance" in row.keys()
                            else 0.0
                        ),
                    )
                return None
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to retrieve word emotion: {e}", {"word_id": word_id}
            ) from e

    @lru_cache(maxsize=256)
    def get_message_emotion(self, message_id: int) -> Optional[MessageEmotionDict]:
        """Retrieve emotional data for a message.

        Args:
            message_id: Database ID of the target message

        Returns:
            Dictionary containing emotion label, confidence and timestamp,
            or None if no data exists for the message

        Raises:
            EmotionError: If the database operation fails

        Example:
            ```python
            emotion_data = emotion_manager.get_message_emotion(1042)
            if emotion_data:
                print(f"Emotion: {emotion_data['label']}")
                print(f"Confidence: {emotion_data['confidence']}")
            ```
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self.config.get_sql_template("get_message_emotion"), (message_id,)
                )
                row: Optional[sqlite3.Row] = cursor.fetchone()

                if row:
                    return MessageEmotionDict(
                        message_id=int(row["message_id"]),
                        label=str(row["label"]),
                        confidence=float(row["confidence"]),
                        timestamp=float(row["timestamp"]),
                    )
                return None
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to retrieve message emotion: {e}", {"message_id": message_id}
            ) from e

    @lru_cache(maxsize=256)
    def _analyze_with_vader(self, input_text: str) -> Tuple[float, float]:
        """Analyze text sentiment using VADER.

        VADER is specifically tuned for social media content and handles
        emoticons, slang, and capitalization particularly well.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal) values from VADER

        Note:
            Returns (0.0, 0.0) if VADER is not available
        """
        if not self.vader:
            return 0.0, 0.0

        input_text = input_text.strip()

        # Get sentiment scores from VADER analyzer
        raw_scores = self.vader.polarity_scores(input_text)  # type: ignore
        vader_scores: VaderSentimentScores = cast(VaderSentimentScores, raw_scores)

        # Map VADER compound score (-1 to 1) directly to valence
        valence = vader_scores["compound"]

        # Estimate arousal from VADER's intensity measures
        # Higher intensity (positive + negative) suggests higher arousal
        intensity = vader_scores["pos"] + vader_scores["neg"]
        arousal = min(intensity * 1.5, 1.0)  # Scale and cap to our range

        return valence, arousal

    @lru_cache(maxsize=256)
    def _analyze_with_llm(self, text: str) -> Tuple[float, float, Dict[str, float]]:
        """Analyze text sentiment using LLM for deeper emotional understanding.

        Uses structured prompting of the language model to extract nuanced
        emotional dimensions beyond basic sentiment, including emotional
        intensity, specific emotional categories, and emotional undertones.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal, emotion_dimensions) where emotion_dimensions
            is a dictionary of additional emotional attributes with their strengths

        Note:
            Returns (0.0, 0.0, {}) if LLM integration is not available
        """
        if not self.llm_interface:
            return 0.0, 0.0, {}

        try:
            # Structured query to the LLM for emotion analysis
            prompt = f"""Analyze the emotional content of this text:
"{text}"

Return the analysis as a JSON object with these fields:
- valence: a float from -1.0 (negative) to 1.0 (positive)
- arousal: a float from 0.0 (calm) to 1.0 (excited)
- primary_emotion: the dominant emotion as a string
- emotion_dimensions: an object with emotional attributes and their strengths (0.0-1.0)
- emotional_undertones: an array of subtle emotional qualities present
"""
            # Get response from LLM and ensure it's a dictionary
            response_raw = self.llm_interface.query(prompt)

            # Process response into dict format - handle both string and dict responses
            response: Dict[str, Any] = {}
            try:
                if response_raw is not None:
                    response = cast(Dict[str, Any], json.loads(response_raw))
            except json.JSONDecodeError as decode_error:
                # Log the invalid JSON input for debugging
                logger.error(
                    f"Invalid JSON response: {response_raw}. Error: {decode_error}"
                )

            # If not valid JSON, create minimal response
            response = {}

            # Extract the core emotional dimensions with safe type handling
            valence_raw: Any = response.get("valence", 0.0)
            try:
                valence = float(valence_raw)
            except (ValueError, TypeError):
                valence = 0.0

            arousal_raw: Any = response.get("arousal", 0.0)
            try:
                arousal = float(arousal_raw)
            except (ValueError, TypeError):
                arousal = 0.0

            # Normalized to expected ranges
            valence = max(self.VALENCE_RANGE[0], min(self.VALENCE_RANGE[1], valence))
            arousal = max(self.AROUSAL_RANGE[0], min(self.AROUSAL_RANGE[1], arousal))

            # Additional emotional dimensions
            emotion_dimensions_raw: Any = response.get("emotion_dimensions", {})
            emotion_dimensions: Dict[str, float] = {}

            # Ensure emotion_dimensions is a valid dictionary with float values
            if isinstance(emotion_dimensions_raw, dict):
                typed_dimensions = cast(Dict[Any, Any], emotion_dimensions_raw)
                for key_raw, value_raw in typed_dimensions.items():
                    key = str(key_raw)
                    try:
                        emotion_dimensions[key] = float(value_raw)
                    except (ValueError, TypeError):
                        emotion_dimensions[key] = 0.0

            return valence, arousal, emotion_dimensions

        except Exception as e:
            # Log error but don't fail - continue with other analyzers
            print(f"LLM emotion analysis failed (continuing with other methods): {e}")
            return 0.0, 0.0, {}

    @lru_cache(maxsize=256)
    def analyze_text_emotion(self, text: str) -> Tuple[float, float]:
        """Analyze text to determine emotional valence and arousal.

        Uses a hybrid approach combining TextBlob, VADER (when available),
        and LLM (when available) for comprehensive sentiment analysis across
        different text styles. Results are weighted and combined according to
        configuration parameters.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal) values normalized to proper ranges

        Example:
            ```python
            valence, arousal = emotion_manager.analyze_text_emotion(
                "I'm feeling really excited about this new project!"
            )
            print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
            # Might output: Valence: 0.82, Arousal: 0.75
            ```
        """

        # Define type for TextBlob sentiment
        class TextBlobSentiment(NamedTuple):
            polarity: float
            subjectivity: float

        # TextBlob analysis (when available)
        if TextBlob is None:
            textblob_valence = 0.0
            subjectivity = 0.0
        else:
            blob = TextBlob(text)
            # Extract sentiment properties safely
            try:
                # Extract properties directly with safe fallbacks
                polarity = getattr(blob.sentiment, "polarity", 0.0)  # type: ignore
                subjectivity = getattr(blob.sentiment, "subjectivity", 0.0)  # type: ignore

                # Create a properly typed TextBlobSentiment
                sentiment_value = TextBlobSentiment(
                    polarity=float(polarity), subjectivity=float(subjectivity)
                )

                # Now we have a properly typed sentiment object
                textblob_valence = sentiment_value.polarity
                subjectivity = sentiment_value.subjectivity
            except (AttributeError, TypeError, ValueError):
                textblob_valence = 0.0
                subjectivity = 0.0

        exclamation_count = text.count("!")
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Initialize weighted results
        valence_components: List[Tuple[float, float]] = [
            (textblob_valence, self.textblob_weight)
        ]
        arousal_factors: List[Tuple[float, float]] = [
            (subjectivity, 1.0),
            (min(exclamation_count / 5, 1.0), 1.0),
            (uppercase_ratio * 2, 1.0),
        ]

        # VADER analysis (when available)
        if self.vader:
            vader_valence, vader_arousal = self._analyze_with_vader(text)
            valence_components.append((vader_valence, self.vader_weight))
            arousal_factors.append(
                (vader_arousal * 2, 2.0)
            )  # VADER's arousal weighted higher

        # LLM analysis (when available)
        if self.llm_interface and self.llm_weight > 0:
            llm_valence, llm_arousal, _ = self._analyze_with_llm(text)
            valence_components.append((llm_valence, self.llm_weight))
            arousal_factors.append((llm_arousal, self.llm_weight * 1.5))

        # Weighted fusion of all analyzers
        total_valence_weight = sum(weight for _, weight in valence_components)
        if total_valence_weight > 0:
            valence = (
                sum(val * weight for val, weight in valence_components)
                / total_valence_weight
            )
        else:
            valence = textblob_valence  # Fallback

        # Combine all arousal factors with their respective weights
        total_arousal_weight = sum(weight for _, weight in arousal_factors)
        if total_arousal_weight > 0:
            arousal = (
                sum(val * weight for val, weight in arousal_factors)
                / total_arousal_weight
            )
        else:
            # Original arousal calculation as fallback
            arousal = sum(val for val, _ in arousal_factors) / len(arousal_factors)

        # Ensure values are within bounds
        valence, arousal = self._clamp_emotional_values(valence, arousal)

        return valence, arousal

    def _count_keyword_occurrences(self, text: str) -> Dict[EmotionCategory, int]:
        """Count emotion keyword occurrences in a text.

        Searches for emotion-specific keywords defined in the configuration,
        using case-insensitive matching to maximize detection. Uses the
        keyword registry for dynamic keyword management.

        Args:
            text: Text to analyze for emotion keywords

        Returns:
            Dictionary mapping emotion categories to keyword occurrence counts

        Example:
            ```python
            counts = emotion_manager._count_keyword_occurrences(
                "I'm feeling happy and excited today!"
            )
            # Might return: {EmotionCategory.HAPPINESS: 2, EmotionCategory.ANGER: 0, ...}
            ```
        """
        text_lower = text.lower()
        result: Dict[EmotionCategory, int] = {}

        # Count occurrences for each emotion category
        for category in EmotionCategory:
            # Get keywords from registry or config
            keywords = self.config.get_keywords_for_emotion(category)
            # Count keyword occurrences
            result[category] = sum(text_lower.count(word) for word in keywords)

        return result

    def classify_emotion(self, text: str) -> Tuple[str, float]:
        """Classify text into basic emotion categories.

        Uses a hybrid approach combining sentiment analysis, keyword detection,
        and optional LLM classification to determine the most likely emotion
        expressed in the text, along with a confidence score.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (emotion_label, confidence)

        Example:
            ```python
            emotion, confidence = emotion_manager.classify_emotion(
                "I'm absolutely thrilled about the news!"
            )
            print(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
            # Might output: Emotion: happiness, Confidence: 0.85
            ```
        """
        # Try LLM classification if available
        if self.llm_interface and self.llm_weight > 0:
            try:
                _, _, emotion_dims = self._analyze_with_llm(text)
                if emotion_dims and "primary_emotion" in emotion_dims:
                    primary = str(emotion_dims["primary_emotion"])
                    confidence = emotion_dims.get("confidence", 0.7)

                    # Try to map to our standard emotion categories
                    try:
                        emotion_cat = EmotionCategory.from_label(primary)
                        return emotion_cat.label, min(confidence, 1.0)
                    except ValueError:
                        # If direct mapping fails, continue with regular classification
                        pass
            except Exception:
                # Continue with regular classification if LLM fails
                pass

        # Get valence for sentiment direction
        valence, _ = self.analyze_text_emotion(text)

        # Count keyword occurrences for each emotion category
        keyword_counts = self._count_keyword_occurrences(text)

        # Identify the category with the most keyword matches
        top_category = max(keyword_counts.items(), key=lambda x: x[1])[0]
        top_count = keyword_counts[top_category]

        # If no keywords found, determine category based on valence
        if top_count == 0:
            if valence > 0.3:
                # Positive valence - choose happiness
                top_category = EmotionCategory.HAPPINESS
            elif valence < -0.3:
                # Negative valence - choose sadness
                top_category = EmotionCategory.SADNESS
            else:
                # Neutral valence - choose neutral
                top_category = EmotionCategory.NEUTRAL
        else:
            # Check if valence agrees with the emotion's typical valence
            is_positive = top_category in POSITIVE_EMOTIONS
            is_negative = top_category in NEGATIVE_EMOTIONS

            if (is_positive and valence < -0.3) or (is_negative and valence > 0.3):
                # Significant disagreement between keyword and valence
                # Choose based more on valence than keywords
                if valence > 0.3:
                    candidates = {c: keyword_counts[c] for c in POSITIVE_EMOTIONS}
                    if any(candidates.values()):
                        # Choose positive emotion with highest keyword count
                        top_category = max(candidates.items(), key=lambda x: x[1])[0]
                    else:
                        # Default to happiness
                        top_category = EmotionCategory.HAPPINESS
                elif valence < -0.3:
                    candidates = {c: keyword_counts[c] for c in NEGATIVE_EMOTIONS}
                    if any(candidates.values()):
                        # Choose negative emotion with highest keyword count
                        top_category = max(candidates.items(), key=lambda x: x[1])[0]
                    else:
                        # Default to sadness
                        top_category = EmotionCategory.SADNESS

        # Calculate confidence based on keyword strength and valence agreement
        # Start with the minimum confidence baseline
        confidence = self.config.min_keyword_confidence

        # Add confidence based on keyword matches (if any)
        if top_count > 0:
            # Use category-specific threshold if available
            keyword_weight = self.config.get_category_weight(top_category)
            keyword_confidence = min(top_count / 3, 1.0) * keyword_weight
            confidence = max(confidence, keyword_confidence)

        # Boost confidence if valence strongly agrees with emotion type
        is_positive_emotion = top_category in POSITIVE_EMOTIONS
        is_negative_emotion = top_category in NEGATIVE_EMOTIONS

        valence_matches_emotion = (valence > 0.3 and is_positive_emotion) or (
            valence < -0.3 and is_negative_emotion
        )

        if valence_matches_emotion:
            confidence = min(confidence + 0.15, 1.0)

        # Use category-specific threshold to potentially adjust confidence
        threshold = top_category.threshold
        if confidence < threshold:
            confidence = (confidence + threshold) / 2  # Blend for smoother transitions

        # Ensure confidence is in valid range
        confidence = max(min(confidence, 1.0), 0.0)

        return top_category.label, confidence

    def _store_message_emotion(
        self, message_id: int, emotion_label: str, confidence: float
    ) -> None:
        """Store emotional data for a message.

        Internal method to handle database insertion or update for message emotions.

        Args:
            message_id: Database ID of the target message
            emotion_label: Text label for the identified emotion
            confidence: Confidence score for the classification (0.0 to 1.0)

        Raises:
            EmotionError: If the database operation fails
        """
        # Clamp confidence value
        confidence = max(
            self.CONFIDENCE_RANGE[0], min(self.CONFIDENCE_RANGE[1], confidence)
        )

        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self.config.get_sql_template("insert_message_emotion"),
                    (message_id, emotion_label, confidence, time.time()),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to store message emotion: {e}",
                {
                    "message_id": message_id,
                    "label": emotion_label,
                    "confidence": confidence,
                },
            ) from e

    def process_message(
        self,
        message_id: int,
        text: str,
        ground_truth: Optional[EmotionCategoryType] = None,
    ) -> EmotionAnalysisDict:
        """Process a message to identify and store its emotional content.

        Performs emotion classification on the message text and stores
        the results in the database for later retrieval and analysis.
        Optionally tracks performance metrics if ground truth is provided.

        Args:
            message_id: Database ID of the message
            text: Message text to analyze
            ground_truth: Optional known emotion for performance tracking

        Returns:
            Dictionary containing the identified emotion data

        Raises:
            EmotionError: If storing the emotion data fails

        Example:
            ```python
            result = emotion_manager.process_message(
                1042, "I'm really disappointed with the service."
            )
            print(f"Detected emotion: {result['emotion_label']}")
            print(f"Confidence: {result['confidence']}")

            # With ground truth for performance tracking
            result = emotion_manager.process_message(
                1043, "I'm so excited!", EmotionCategory.HAPPINESS
            )
            ```
        """
        # Classify the emotion in the text
        emotion_label, confidence = self.classify_emotion(text)

        # Convert string emotion label to EmotionCategory for internal use
        try:
            predicted_emotion = EmotionCategory.from_label(emotion_label)
        except ValueError:
            # Fallback if label doesn't match any category
            predicted_emotion = EmotionCategory.NEUTRAL
            emotion_label = predicted_emotion.label

        # Store the emotion data using the correct private method
        self._store_message_emotion(message_id, emotion_label, confidence)

        # Track metrics if ground_truth is provided
        if ground_truth is not None:
            try:
                actual_emotion = normalize_emotion_category(ground_truth)
                # Record detection for performance tracking
                self.metrics.record_detection(predicted_emotion, actual_emotion)

                # Increment detection count
                self._detection_count += 1

                # Periodically optimize weights if enough data
                if self._detection_count % self._optimization_frequency == 0:
                    optimized_weights = self.metrics.optimize_weights(self.config)
                    self.config.per_category_weights.update(optimized_weights)
            except ValueError:
                # Skip metrics if ground truth is invalid
                pass

        return {"emotion_label": emotion_label, "confidence": confidence}

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get a report of emotion detection performance metrics.

        Summarizes the performance of emotion detection across categories,
        useful for monitoring system accuracy and identifying areas for
        improvement.

        Returns:
            Dictionary with performance metrics for each emotion category

        Example:
            ```python
            report = emotion_manager.get_metrics_report()
            print(f"Overall accuracy: {report['overall_accuracy']:.2f}")
            for category, metrics in report['categories'].items():
                print(f"{category}: precision={metrics['precision']:.2f}, "
                      f"recall={metrics['recall']:.2f}, f1={metrics['f1']:.2f}")
            ```
        """
        # Skip report if no detections recorded
        if self.metrics.total_detections == 0:
            return {"total_detections": 0, "overall_accuracy": 0.0, "categories": {}}

        # Calculate metrics for each category
        category_metrics = {}
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for category in EmotionCategory:
            precision = self.metrics.get_precision(category)
            recall = self.metrics.get_recall(category)
            f1 = self.metrics.get_f1_score(category)

            category_metrics[category.label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": self.metrics.true_positives.get(category, 0),
                "false_positives": self.metrics.false_positives.get(category, 0),
                "false_negatives": self.metrics.false_negatives.get(category, 0),
            }

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        # Calculate overall metrics
        category_count = len(EmotionCategory)

        return {
            "total_detections": self.metrics.total_detections,
            "overall_accuracy": (
                sum(self.metrics.true_positives.values())
                / self.metrics.total_detections
                if self.metrics.total_detections > 0
                else 0.0
            ),
            "avg_precision": total_precision / category_count,
            "avg_recall": total_recall / category_count,
            "avg_f1": total_f1 / category_count,
            "categories": category_metrics,
            "optimization_count": self._detection_count // self._optimization_frequency,
        }

    def analyze_term_recursively(
        self, term: str, context: Optional[Union[str, EmotionalContext]] = None
    ) -> FullEmotionAnalysisDict:
        """Perform deep emotional analysis on a term using recursive processing.

        This method provides a comprehensive emotional analysis beyond basic
        sentiment analysis, including meta-emotions, emotional patterns, and
        contextual emotional responses. Uses a recursive model that explores
        emotional associations at multiple levels.

        Args:
            term: The word or phrase to analyze
            context: Optional emotional context or context name

        Returns:
            Dictionary containing comprehensive emotional analysis including:
            - Primary emotion classification
            - Dimensional emotional measurements
            - Meta-emotions (emotions about emotions)
            - Emotional patterns and sequences
            - Recursive analysis depth

        Note:
            This is a more computationally intensive analysis than basic
            sentiment classification and should be used selectively.

        Example:
            ```python
            deep_analysis = emotion_manager.analyze_term_recursively("betrayal")
            print(f"Primary emotion: {deep_analysis['emotion_label']}")
            print(f"Dimensions: {deep_analysis['dimensions']}")
            print(f"Meta-emotions: {[m['label'] for m in deep_analysis['meta_emotions']]}")
            ```
        """
        # Prepare context if provided as a string
        actual_context = None
        if isinstance(context, str):
            actual_context = self.recursive_processor.get_context(context)
            if not actual_context and context:
                # Try to create a domain-specific context
                actual_context = self.recursive_processor.create_context_for_domain(
                    context
                )
        elif isinstance(context, EmotionalContext):
            actual_context = context

        # Process the term using recursive processor
        concept = self.recursive_processor.process_term(term, actual_context)

        # Get the basic emotion category and confidence
        emotion_label, confidence = self.classify_emotion(term)

        # Enhance with LLM-derived insights if available
        additional_insights: Dict[str, Any] = {}
        if self.llm_interface and self.llm_weight > 0:
            try:
                prompt = f"""Analyze the emotional associations of the term '{term}' in depth.

Focus especially on:
1. Meta-emotions (emotions about emotions)
2. Emotional patterns or sequences this term might evoke
3. Cultural and contextual emotional variations
4. Temporal dynamics (how emotions related to this term might change over time)

Format your response as a structured JSON with insights in each category."""

                insights = self.llm_interface.query(prompt)
                if isinstance(insights, (dict, str)) and insights:
                    additional_insights = insights if isinstance(insights, dict) else {}
            except Exception as e:
                # Continue without LLM insights if there's an error
                print(f"LLM insights generation failed: {e}")

        # Create the return structure
        result: FullEmotionAnalysisDict = {
            "emotion_label": emotion_label,
            "confidence": confidence,
            "concept": concept.as_dict(),
            "dimensions": concept.primary_emotion.as_dict(),
            "meta_emotions": [
                {"label": label, "dimensions": emotion.as_dict()}
                for label, emotion in concept.meta_emotions
            ],
            "patterns": {
                pattern_type: [emotion.as_dict() for emotion in sequence]
                for pattern_type, sequence in concept.emotional_patterns.items()
            },
            "recursive_depth": concept.recursive_depth,
        }

        # Add LLM insights if available
        if additional_insights:
            result["enhanced_insights"] = additional_insights

        return cast(FullEmotionAnalysisDict, result)

    def analyze_emotional_relationship(
        self, term1: str, term2: str, relationship_type: str
    ) -> float:
        """Analyze the emotional relationship between two terms.

        Evaluates the strength and nature of emotional relationships between
        concepts, allowing for sophisticated emotional reasoning.

        Args:
            term1: First term to analyze
            term2: Second term to analyze
            relationship_type: Type of relationship to assess

        Returns:
            Strength of the relationship (0.0-1.0)

        Relationship types include:
        - emotional_synonym: Similar emotional quality
        - emotional_antonym: Opposite emotional quality
        - intensifies/diminishes: One term modifies intensity
        - emotional_component/composite: Part-whole relationships
        - evokes: One term triggers the emotion of another
        - meta_emotion: Emotion about an emotion

        Example:
            ```python
            # Check if "ecstatic" intensifies "happy"
            strength = emotion_manager.analyze_emotional_relationship(
                "ecstatic", "happy", "intensifies"
            )
            print(f"Intensification strength: {strength:.2f}")
            # Might output: Intensification strength: 0.87
            ```
        """
        # Use enhanced LLM analysis if available for more nuanced relationship assessment
        if self.llm_interface and self.llm_weight > 0:
            try:
                prompt = f"""Analyze the emotional relationship between '{term1}' and '{term2}'
specifically focusing on the relationship type: '{relationship_type}'.

Assess the strength of this relationship on a scale from 0.0 (no relationship) to 1.0 (strongest possible relationship).

Consider:
- Semantic similarity and difference
- Emotional intensity differences
- Cultural and contextual factors
- Linguistic patterns of usage

Return only a numeric value between 0.0 and 1.0 representing the strength."""

                result = self.llm_interface.query(prompt)

                # Get base strength from recursive processor for fallback or blending
                base_strength = self.recursive_processor.analyze_relationship(
                    term1, term2, relationship_type
                )

                # Extract numeric value from result
                if result is not None:
                    try:
                        # Ensure result is a string before processing
                        strength = float(result.strip())
                        # Ensure proper range
                        strength = max(0.0, min(1.0, strength))

                        # Weighted average
                        return strength * self.llm_weight + base_strength * (
                            1 - self.llm_weight
                        )
                    except (ValueError, TypeError):
                        # If conversion fails, fall back to base strength
                        return base_strength
            except Exception:
                # Fall back to recursive processor if LLM fails
                pass

        # Default recursive processor analysis
        return self.recursive_processor.analyze_relationship(
            term1, term2, relationship_type
        )

    def create_emotional_context(
        self,
        domain: Optional[str] = None,
        cultural_factors: Optional[Dict[str, float]] = None,
        situational_factors: Optional[Dict[str, float]] = None,
        temporal_factors: Optional[Dict[str, float]] = None,
        domain_specific: Optional[Dict[str, float]] = None,
    ) -> EmotionalContext:
        """Create a custom emotional context for analysis.

        Emotional contexts provide frameworks for interpreting emotions
        within specific domains, cultures, or situations. They adjust the
        baseline emotional responses to account for contextual factors.

        Args:
            domain: Optional domain name for predefined contexts
            cultural_factors: Optional cultural dimension adjustments
            situational_factors: Optional situational adjustments
            temporal_factors: Optional temporal dimension adjustments
            domain_specific: Optional domain-specific adjustments

        Returns:
            An emotional context object for use in recursive analysis

        Example:
            ```python
            # Create a context for analyzing medical terminology
            medical_context = emotion_manager.create_emotional_context(
                domain="medical",
                situational_factors={"clinical_setting": 0.8, "emergency": 0.2}
            )

            # Analyze term within this context
            analysis = emotion_manager.analyze_term_recursively(
                "diagnosis", medical_context
            )
            ```
        """
        # If domain is provided, start with domain-specific context
        if domain:
            context = self.recursive_processor.create_context_for_domain(domain)
        else:
            context = EmotionalContext()

        # Apply any provided factor dictionaries
        if cultural_factors:
            context.cultural_factors.update(cultural_factors)
        if situational_factors:
            context.situational_factors.update(situational_factors)
        if temporal_factors:
            context.temporal_factors.update(temporal_factors)
        if domain_specific:
            context.domain_specific.update(domain_specific)

        return context

    def register_emotional_context(self, name: str, context: EmotionalContext) -> None:
        """Register a named emotional context for reuse.

        Stores a context in the system's registry for later retrieval,
        allowing consistent application of the same contextual factors
        across multiple analyses.

        Args:
            name: Name to identify this context
            context: The emotional context to register

        Example:
            ```python
            # Create and register a "clinical" context
            clinical = emotion_manager.create_emotional_context(
                domain="medical",
                situational_factors={"professional": 0.9, "emergency": 0.1}
            )
            emotion_manager.register_emotional_context("clinical", clinical)

            # Later retrieve by name
            analysis = emotion_manager.analyze_term_recursively(
                "prognosis", "clinical"  # Using string name to reference context
            )
            ```
        """
        self.recursive_processor.register_context(name, context)

    def get_emotional_context(self, name: str) -> Optional[EmotionalContext]:
        """Retrieve a registered emotional context by name.

        Args:
            name: Name of the context to retrieve

        Returns:
            The emotional context if found, None otherwise

        Example:
            ```python
            # Retrieve a previously registered context
            clinical_context = emotion_manager.get_emotional_context("clinical")
            if clinical_context:
                # Use the context
                emotion_manager.analyze_term_recursively("diagnosis", clinical_context)
            ```
        """
        return self.recursive_processor.get_context(name)

    def enrich_word_emotions(self, term: str, word_id: int) -> bool:
        """Enrich emotional data for a word using all available analysis methods.

        Combines basic sentiment analysis with deeper LLM-based insights when
        available to provide comprehensive emotional profiling of a term.

        Args:
            term: The word to analyze
            word_id: Database ID of the word

        Returns:
            Boolean indicating whether enrichment was successful

        Example:
            ```python
            word_id = db_manager.get_word_id("nostalgia")
            emotion_manager.enrich_word_emotions("nostalgia", word_id)
            # Retrieves and stores enriched emotional data for "nostalgia"
            ```
        """
        try:
            # Basic sentiment analysis
            valence, arousal = self.analyze_text_emotion(term)

            # If LLM is available, get deeper insights
            if self.llm_interface and self.llm_weight > 0:
                try:
                    prompt = f"""Analyze the emotional qualities of the word '{term}'.

Provide a detailed analysis of:
1. Valence (positivity/negativity) on a scale from -1.0 to 1.0
2. Arousal (intensity/energy) on a scale from 0.0 to 1.0
3. The complexity and nuance of emotions typically associated with this term
4. Common contexts where this word carries emotional weight
5. Cultural variations in emotional perception of this term

Focus on emotional qualities rather than definitions."""

                    insights = self.llm_interface.query(prompt)

                    # Extract more precise valence/arousal from LLM if possible
                    if insights is not None:
                        llm_valence, llm_arousal, _ = self._analyze_with_llm(
                            term + ". " + str(insights)
                        )

                        # Blend basic and LLM-enhanced values
                        if isinstance(llm_valence, float) and isinstance(
                            llm_arousal, float
                        ):
                            valence = (valence + llm_valence) / 2
                            arousal = (arousal + llm_arousal) / 2

                except Exception as e:
                    # Continue with basic analysis if LLM fails
                    print(f"LLM word enrichment failed for '{term}': {e}")

            # Store the enriched emotional data
            self.set_word_emotion(word_id, valence, arousal)

            # If we have recursive processing capability, do deeper analysis
            if self._recursive_processor:
                # Just trigger the processing - results are cached internally
                self.analyze_term_recursively(term)

            return True

        except Exception as e:
            print(f"Error enriching word emotions for '{term}': {e}")
            return False

    def _store_emotional_relationship(
        self,
        cursor: sqlite3.Cursor,
        word_id: int,
        anchor_term: str,
        relationship_type: str,
        valence: float,
        arousal: float,
        timestamp: float,
    ) -> None:
        """Persist the derived emotional relationship for a word."""

        try:
            cursor.execute(
                SQL_UPSERT_EMOTIONAL_RELATIONSHIP,
                (word_id, anchor_term, relationship_type, valence, arousal, timestamp),
            )
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to store emotional relationship for word_id {word_id}: {e}",
                {
                    "word_id": word_id,
                    "related_term": anchor_term,
                    "relationship_type": relationship_type,
                },
            ) from e

    def _derive_emotional_relationship(
        self, valence: float, arousal: float
    ) -> Tuple[str, str]:
        """Map valence/arousal readings to a relationship type and anchor term."""

        if valence >= 0.35:
            if arousal >= 0.6:
                return "joy_associated", "joy"
            return "trust_associated", "trust"
        if valence <= -0.35:
            if arousal >= 0.6:
                return "anger_associated", "anger"
            return "sadness_associated", "sadness"
        if arousal >= 0.6:
            return "emotionally_charged", "excitement"
        return "emotional_neutral", "calm"

    def _ensure_emotion_anchor(self, term: str) -> None:
        """Ensure the anchor term exists in the lexical database."""

        if not term:
            return

        definition = f"Emotion anchor node representing '{term}'."
        try:
            self.db_manager.insert_or_update_word(
                term,
                definition=definition,
                part_of_speech="noun",
                usage_examples=None,
            )
        except Exception as exc:
            raise EmotionError(
                f"Failed to ensure emotion anchor '{term}': {exc}",
                {"anchor_term": term},
            ) from exc
