"""
This module provides a comprehensive configuration framework for emotion analysis operations
in the Word Forge system. It centralizes parameters, constraints, and database schemas
for processing and storing emotion-related data.

Key components:
- EmotionConfig: Main configuration dataclass that controls all emotion analysis parameters
- Validation utilities for ensuring emotion measurements stay within valid ranges
- SQL templates for database operations on emotion data
- Type definitions for emotion categories, keywords, and validation results

Features:
- Sentiment analysis configuration (VADER and TextBlob integrations)
- Valence-arousal model implementation for dimensional emotion representation
- Categorical emotion classification with keyword-based detection
- Database schema definitions for persistent emotion storage
- Environment variable mapping for runtime configuration
- Dynamic keyword registry with multiple data source support
- SQL dialect adapters for database portability
- Self-optimizing emotion detection through performance metrics

Usage examples:
    # Create default configuration
    config = EmotionConfig()

    # Validate emotion measurements
    if config.is_valid_valence(0.75):
        # Process valid valence value
        pass

    # Get detailed validation information
    result = config.validate_arousal(1.2)
    if not result.is_valid:
        logger.warning(result.message)  # "Arousal value 1.2 is outside valid range (0.0, 1.0)"

    # Access SQL templates for database operations
    db.execute(config.get_sql_template("create_word_emotion_table"))

    # Get keywords associated with an emotion
    happiness_keywords = config.get_keywords_for_emotion(EmotionCategory.HAPPINESS)

    # Load additional keywords from external source
    config.keyword_registry.load_from_json("custom_emotions.json")

    # Record detection results for optimization
    metrics = EmotionDetectionMetrics()
    metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)  # True positive

    # Optimize weights based on detection performance
    optimized_weights = metrics.optimize_weights(config)
    config.per_category_weights.update(optimized_weights)

    The emotion system uses a hybrid approach combining:
    - Dimensional model: Valence (positive/negative) and arousal (intensity)
    - Categorical model: Discrete emotion classes (happiness, sadness, etc.)
    - Lexical approach: Keyword-based emotion detection
    - ML integration: VADER and TextBlob sentiment analysis

Dependencies:
    - word_forge.configs.config_essentials: For EmotionRange and EnvMapping types
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from math import isclose
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    List,
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    overload,
)

from word_forge.configs.config_essentials import EmotionRange, EnvMapping

# Legacy type definition for backward compatibility
EmotionCategoryLiteral = Literal[
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]


# Modern Enum-based implementation
class EmotionCategory(Enum):
    """
    Enumeration of supported emotion categories with associated metadata.

    Each emotion category has a string label, weight, and detection threshold.
    These values can be used to fine-tune emotion classification algorithms.

    Attributes:
        label: String representation of the emotion
        weight: Default weight for this emotion in classification algorithms
        threshold: Minimum confidence threshold for detection
    """

    HAPPINESS = ("happiness", 0.8, 0.7)
    SADNESS = ("sadness", 0.7, 0.6)
    ANGER = ("anger", 0.8, 0.8)
    FEAR = ("fear", 0.7, 0.7)
    SURPRISE = ("surprise", 0.6, 0.5)
    DISGUST = ("disgust", 0.7, 0.7)
    NEUTRAL = ("neutral", 0.3, 0.3)

    def __init__(self, label: str, weight: float, threshold: float):
        self.label = label
        self.weight = weight  # Category-specific weight
        self.threshold = threshold  # Detection threshold

    @classmethod
    def from_label(cls, label: str) -> "EmotionCategory":
        """
        Convert string label to enum value.

        Args:
            label: String representation of the emotion

        Returns:
            The corresponding EmotionCategory enum value

        Raises:
            ValueError: If the label doesn't match any known emotion category

        Examples:
            >>> EmotionCategory.from_label("happiness")
            EmotionCategory.HAPPINESS
            >>> EmotionCategory.from_label("unknown")
            Traceback (most recent call last):
            ...
            ValueError: Unknown emotion category: unknown
        """
        for emotion in cls:
            if emotion.label == label:
                return emotion
        raise ValueError(f"Unknown emotion category: {label}")

    def __str__(self) -> str:
        """Return the string label of the emotion category."""
        return self.label


# Type conversion helper for backward compatibility
EmotionCategoryType = Union[EmotionCategory, EmotionCategoryLiteral]


def normalize_emotion_category(category: EmotionCategoryType) -> EmotionCategory:
    """
    Convert any emotion category representation to an EmotionCategory enum.

    Handles both string literals and EmotionCategory enum values for
    backward compatibility.

    Args:
        category: String label or EmotionCategory enum

    Returns:
        EmotionCategory enum value

    Raises:
        ValueError: If the category can't be converted to a valid EmotionCategory
    """
    if isinstance(category, EmotionCategory):
        return category
    try:
        return EmotionCategory.from_label(category)
    except ValueError as e:
        raise ValueError(f"Invalid emotion category: {category}") from e


T = TypeVar("T")  # Generic type for validation results


class EmotionValidationResult(NamedTuple):
    """Result of emotion value validation with detailed context."""

    is_valid: bool
    """Whether the value is valid according to configuration constraints."""

    message: str
    """Descriptive message explaining validation result."""

    value: float
    """The value that was validated."""

    range: EmotionRange
    """The range against which the value was validated."""


class SQLTemplateProtocol(Protocol):
    """Protocol defining the SQL template interface for emotion operations."""

    @property
    def create_word_emotion_table(self) -> str: ...

    @property
    def create_message_emotion_table(self) -> str: ...

    @property
    def insert_word_emotion(self) -> str: ...

    @property
    def get_word_emotion(self) -> str: ...

    @property
    def insert_message_emotion(self) -> str: ...

    @property
    def get_message_emotion(self) -> str: ...


class EmotionKeywordsDict(TypedDict):
    """Type definition for the emotion keywords dictionary."""

    happiness: List[str]
    sadness: List[str]
    anger: List[str]
    fear: List[str]
    surprise: List[str]
    disgust: List[str]
    neutral: List[str]


# Define a TypedDict for VADER sentiment scores
class VaderSentimentScores(TypedDict):
    """Type definition for VADER sentiment analyzer output."""

    pos: float
    neg: float
    neu: float
    compound: float


# ==========================================
# SQL Template Constants
# ==========================================
# Exported at module level for backward compatibility

#: SQL schema for word emotion table
SQL_CREATE_WORD_EMOTION_TABLE: Final[
    str
] = """
    CREATE TABLE IF NOT EXISTS word_emotion (
        word_id INTEGER PRIMARY KEY,
        valence REAL NOT NULL,
        arousal REAL NOT NULL,
        timestamp REAL NOT NULL,
        FOREIGN KEY(word_id) REFERENCES words(id)
    );
"""

#: SQL schema for message emotion table
SQL_CREATE_MESSAGE_EMOTION_TABLE: Final[
    str
] = """
    CREATE TABLE IF NOT EXISTS message_emotion (
        message_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp REAL NOT NULL
    );
"""

#: SQL query for inserting word emotion data
SQL_INSERT_WORD_EMOTION: Final[
    str
] = """
    INSERT OR REPLACE INTO word_emotion
    (word_id, valence, arousal, timestamp)
    VALUES (?, ?, ?, ?)
"""

#: SQL query for retrieving word emotion data
SQL_GET_WORD_EMOTION: Final[
    str
] = """
    SELECT word_id, valence, arousal, timestamp
    FROM word_emotion
    WHERE word_id = ?
"""

#: SQL query for inserting message emotion data
SQL_INSERT_MESSAGE_EMOTION: Final[
    str
] = """
    INSERT OR REPLACE INTO message_emotion
    (message_id, label, confidence, timestamp)
    VALUES (?, ?, ?, ?)
"""

#: SQL query for retrieving message emotion data
SQL_GET_MESSAGE_EMOTION: Final[
    str
] = """
    SELECT message_id, label, confidence, timestamp
    FROM message_emotion
    WHERE message_id = ?
"""


# ==========================================
# SQL Dialect Support
# ==========================================


class SQLDialect(Enum):
    """
    SQL dialect types supported by the system.

    Different database systems use slightly different SQL syntax.
    This enum enables dialect-specific template adaptations.
    """

    SQLITE = auto()
    POSTGRESQL = auto()
    MYSQL = auto()


@dataclass
class EmotionSQLTemplates:
    """
    SQL templates for emotion data operations with dialect support.

    Provides database-agnostic access to SQL templates, automatically
    adapting the syntax to match the configured dialect.

    Attributes:
        dialect: The SQL dialect to use for template adaptation
    """

    dialect: SQLDialect = SQLDialect.SQLITE

    def get_template(self, operation: str) -> str:
        """
        Get SQL template for a specific operation with dialect adaptations.

        Args:
            operation: Name of the SQL operation to retrieve

        Returns:
            SQL query string adapted to the configured dialect

        Examples:
            >>> templates = EmotionSQLTemplates(dialect=SQLDialect.POSTGRESQL)
            >>> sql = templates.get_template("create_word_emotion_table")
            >>> "SERIAL PRIMARY KEY" in sql  # PostgreSQL syntax
            True
        """
        template = self._get_base_template(operation)
        return self._adapt_to_dialect(template)

    def _get_base_template(self, operation: str) -> str:
        """
        Get the base SQL template for an operation.

        Args:
            operation: Name of the SQL operation

        Returns:
            Base SQL template string (SQLite syntax)
        """
        templates = {
            "create_word_emotion_table": SQL_CREATE_WORD_EMOTION_TABLE,
            "create_message_emotion_table": SQL_CREATE_MESSAGE_EMOTION_TABLE,
            "insert_word_emotion": SQL_INSERT_WORD_EMOTION,
            "get_word_emotion": SQL_GET_WORD_EMOTION,
            "insert_message_emotion": SQL_INSERT_MESSAGE_EMOTION,
            "get_message_emotion": SQL_GET_MESSAGE_EMOTION,
        }
        return templates.get(operation, "")

    def _adapt_to_dialect(self, template: str) -> str:
        """
        Adapt SQL template to the configured dialect.

        Args:
            template: Base SQL template (SQLite syntax)

        Returns:
            SQL template adapted to the configured dialect
        """
        if self.dialect == SQLDialect.SQLITE:
            return template
        elif self.dialect == SQLDialect.POSTGRESQL:
            # Convert SQLite syntax to PostgreSQL
            return template.replace("INTEGER PRIMARY KEY", "SERIAL PRIMARY KEY")
        elif self.dialect == SQLDialect.MYSQL:
            # Convert SQLite syntax to MySQL
            return template.replace(
                "INTEGER PRIMARY KEY", "INT AUTO_INCREMENT PRIMARY KEY"
            )
        return template


# ==========================================
# Emotion Keyword Registry
# ==========================================


@dataclass
class EmotionKeywordRegistry:
    """
    Registry for managing emotion keywords with dynamic loading capabilities.

    Provides a centralized repository for emotion-related keywords that can be
    loaded from multiple sources and accessed by emotion category.

    Attributes:
        _keywords: Dictionary mapping emotion categories to keyword lists
        _sources: List of data sources that have been loaded
    """

    _keywords: Dict[EmotionCategory, List[str]] = field(default_factory=dict)
    _sources: List[str] = field(default_factory=list)

    def load_from_json(self, path: str) -> None:
        """
        Load emotion keywords from a JSON file.

        The JSON file should have a structure like:
        {
            "happiness": ["joyful", "elated", ...],
            "sadness": ["miserable", "depressed", ...],
            ...
        }

        Args:
            path: Path to the JSON file

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            json.JSONDecodeError: If the JSON file is invalid
        """
        import json

        with open(path, "r") as f:
            data = json.load(f)
            self._update_from_dict(data)
            self._sources.append(path)

    def load_from_database(self, connection: Any, query: str) -> None:
        """
        Load emotion keywords from a database.

        Args:
            connection: Database connection object
            query: SQL query that returns rows with (emotion, keyword) columns

        Raises:
            Exception: If a database error occurs
        """
        cursor = connection.cursor()
        cursor.execute(query)

        emotion_keywords: Dict[str, List[str]] = {}
        for emotion_label, keyword in cursor.fetchall():
            if emotion_label not in emotion_keywords:
                emotion_keywords[emotion_label] = []
            emotion_keywords[emotion_label].append(keyword)

        self._update_from_dict(emotion_keywords)
        self._sources.append(f"database:{query}")

    @overload
    def get_keywords(self, emotion: EmotionCategory) -> List[str]: ...

    @overload
    def get_keywords(self, emotion: EmotionCategoryLiteral) -> List[str]: ...

    def get_keywords(self, emotion: EmotionCategoryType) -> List[str]:
        """
        Get keywords for a specific emotion category.

        Args:
            emotion: Emotion category (enum or string label)

        Returns:
            List of keywords associated with the emotion

        Examples:
            >>> registry = EmotionKeywordRegistry()
            >>> registry.register_keywords(EmotionCategory.HAPPINESS, ["joyful", "elated"])
            >>> registry.get_keywords(EmotionCategory.HAPPINESS)
            ['joyful', 'elated']
            >>> registry.get_keywords("happiness")  # String label also works
            ['joyful', 'elated']
        """
        category = normalize_emotion_category(emotion)
        return self._keywords.get(category, [])

    @overload
    def register_keywords(
        self, emotion: EmotionCategory, keywords: List[str]
    ) -> None: ...

    @overload
    def register_keywords(
        self, emotion: EmotionCategoryLiteral, keywords: List[str]
    ) -> None: ...

    def register_keywords(
        self, emotion: EmotionCategoryType, keywords: List[str]
    ) -> None:
        """
        Register new keywords for an emotion category.

        Args:
            emotion: Emotion category (enum or string label)
            keywords: List of keywords to associate with the emotion

        Examples:
            >>> registry = EmotionKeywordRegistry()
            >>> registry.register_keywords(EmotionCategory.HAPPINESS, ["joyful", "elated"])
            >>> registry.register_keywords("sadness", ["miserable", "depressed"])
        """
        category = normalize_emotion_category(emotion)
        if category in self._keywords:
            self._keywords[category].extend(
                [k for k in keywords if k not in self._keywords[category]]
            )
        else:
            self._keywords[category] = keywords.copy()

    def _update_from_dict(self, data: Dict[str, List[str]]) -> None:
        """
        Update registry from a dictionary mapping.

        Args:
            data: Dictionary mapping emotion labels to keyword lists
        """
        for label, keywords in data.items():
            try:
                emotion = EmotionCategory.from_label(label)
                self.register_keywords(emotion, keywords)
            except ValueError:
                continue  # Skip unknown emotion categories

    def clear(self) -> None:
        """Clear all registered keywords and sources."""
        self._keywords.clear()
        self._sources.clear()

    def get_sources(self) -> List[str]:
        """Get list of loaded data sources."""
        return self._sources.copy()


# ==========================================
# Emotion Detection Metrics
# ==========================================


@dataclass
class EmotionDetectionMetrics:
    """
    Metrics for tracking and optimizing emotion detection performance.

    Records detection outcomes and provides methods for calculating performance
    metrics and optimizing detection parameters.

    Attributes:
        total_detections: Total number of emotion detections recorded
        true_positives: Count of correct detections by emotion category
        false_positives: Count of incorrect detections by emotion category
    """

    total_detections: int = 0
    true_positives: Dict[EmotionCategory, int] = field(
        default_factory=lambda: {e: 0 for e in EmotionCategory}
    )
    false_positives: Dict[EmotionCategory, int] = field(
        default_factory=lambda: {e: 0 for e in EmotionCategory}
    )
    false_negatives: Dict[EmotionCategory, int] = field(
        default_factory=lambda: {e: 0 for e in EmotionCategory}
    )

    @overload
    def record_detection(
        self, predicted: EmotionCategory, actual: EmotionCategory
    ) -> None: ...

    @overload
    def record_detection(
        self, predicted: EmotionCategoryLiteral, actual: EmotionCategoryLiteral
    ) -> None: ...

    def record_detection(
        self, predicted: EmotionCategoryType, actual: EmotionCategoryType
    ) -> None:
        """
        Record a detection result for optimization metrics.

        Args:
            predicted: Emotion category predicted by the system
            actual: Actual emotion category (ground truth)

        Examples:
            >>> metrics = EmotionDetectionMetrics()
            >>> metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
            >>> metrics.true_positives[EmotionCategory.HAPPINESS]
            1
            >>> metrics.record_detection("anger", "fear")  # String labels also work
            >>> metrics.false_positives[EmotionCategory.ANGER]
            1
            >>> metrics.false_negatives[EmotionCategory.FEAR]
            1
        """
        pred_category = normalize_emotion_category(predicted)
        actual_category = normalize_emotion_category(actual)

        self.total_detections += 1
        if pred_category == actual_category:
            self.true_positives[pred_category] = (
                self.true_positives.get(pred_category, 0) + 1
            )
        else:
            self.false_positives[pred_category] = (
                self.false_positives.get(pred_category, 0) + 1
            )
            self.false_negatives[actual_category] = (
                self.false_negatives.get(actual_category, 0) + 1
            )

    def get_precision(self, category: EmotionCategoryType) -> float:
        """
        Calculate precision for a specific emotion category.

        Precision = true positives / (true positives + false positives)

        Args:
            category: Emotion category to calculate precision for

        Returns:
            Precision value between 0.0 and 1.0

        Examples:
            >>> metrics = EmotionDetectionMetrics()
            >>> metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
            >>> metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.SADNESS)
            >>> metrics.get_precision(EmotionCategory.HAPPINESS)
            0.5
        """
        cat = normalize_emotion_category(category)
        tp = self.true_positives.get(cat, 0)
        fp = self.false_positives.get(cat, 0)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def get_recall(self, category: EmotionCategoryType) -> float:
        """
        Calculate recall for a specific emotion category.

        Recall = true positives / (true positives + false negatives)

        Args:
            category: Emotion category to calculate recall for

        Returns:
            Recall value between 0.0 and 1.0
        """
        cat = normalize_emotion_category(category)
        tp = self.true_positives.get(cat, 0)
        fn = self.false_negatives.get(cat, 0)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def get_f1_score(self, category: EmotionCategoryType) -> float:
        """
        Calculate F1 score for a specific emotion category.

        F1 = 2 * (precision * recall) / (precision + recall)

        Args:
            category: Emotion category to calculate F1 score for

        Returns:
            F1 score between 0.0 and 1.0
        """
        precision = self.get_precision(category)
        recall = self.get_recall(category)
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    def optimize_weights(self, config: "EmotionConfig") -> Dict[EmotionCategory, float]:
        """
        Recursively optimize category weights based on detection performance.

        Adjusts weights higher for high-precision categories and lower for
        low-precision categories to improve overall detection accuracy.

        Args:
            config: Current emotion configuration

        Returns:
            Dictionary mapping emotion categories to optimized weights

        Examples:
            >>> metrics = EmotionDetectionMetrics()
            >>> config = EmotionConfig()
            >>> # Record some detections...
            >>> optimized_weights = metrics.optimize_weights(config)
            >>> config.per_category_weights.update(optimized_weights)
        """
        new_weights: Dict[EmotionCategory, float] = {}

        # If we don't have enough data, return current weights
        if self.total_detections < 10:
            return {cat: config.get_category_weight(cat) for cat in EmotionCategory}

        # Optimize weights based on precision
        for category in EmotionCategory:
            precision = self.get_precision(category)
            current_weight = config.get_category_weight(category)

            # Adjust weight based on precision
            if precision < 0.5:
                # Low precision - decrease weight
                new_weights[category] = max(0.1, current_weight * 0.9)
            elif precision > 0.8:
                # High precision - increase weight
                new_weights[category] = min(1.0, current_weight * 1.1)
            else:
                # Maintain current weight
                new_weights[category] = current_weight

        return new_weights

    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.total_detections = 0
        self.true_positives = {e: 0 for e in EmotionCategory}
        self.false_positives = {e: 0 for e in EmotionCategory}
        self.false_negatives = {e: 0 for e in EmotionCategory}


@dataclass
class EmotionConfig:
    """
    Configuration for emotion analysis operations.

    Controls sentiment analysis parameters, emotion classification rules,
    and database schema for emotion data storage and retrieval.

    Attributes:
        enable_vader: Whether to use VADER for sentiment analysis
        vader_weight: Weight given to VADER in hybrid sentiment analysis
        textblob_weight: Weight given to TextBlob in hybrid sentiment analysis
        valence_range: Valid range for valence values (negative to positive)
        arousal_range: Valid range for arousal values (calm to excited)
        confidence_range: Valid range for confidence levels (certainty)
        sql_templates: Dictionary of SQL templates for emotion data operations
        emotion_keywords: Dictionary mapping emotion categories to keywords
        min_keyword_confidence: Minimum confidence when no keywords found
        keyword_match_weight: Weight given to keyword matches in classification
        keyword_registry: Registry for dynamic emotion keyword management
        sql_dialect: SQL dialect to use for database operations
        per_category_weights: Category-specific weights for classification
        enable_caching: Whether to cache emotion analysis results
        cache_ttl: Time-to-live for cached results in seconds
        language: Language code for language-specific processing
        ENV_VARS: Mapping of environment variables to config attributes

    Examples:
        >>> config = EmotionConfig()
        >>> config.is_valid_valence(0.5)
        True
        >>> config.is_valid_valence(2.0)  # Out of range
        False
        >>> config.get_sql_template("get_word_emotion")  # New method
        'SELECT word_id, valence, arousal, timestamp\\n    FROM word_emotion\\n    WHERE word_id = ?'
        >>> config.SQL_GET_WORD_EMOTION  # Legacy property still works
        'SELECT word_id, valence, arousal, timestamp\\n    FROM word_emotion\\n    WHERE word_id = ?'
    """

    # ==========================================
    # Sentiment Analysis Configuration
    # ==========================================

    #: Whether to use VADER for sentiment analysis
    enable_vader: bool = True

    #: Weight given to VADER in hybrid sentiment analysis
    vader_weight: float = 0.7

    #: Weight given to TextBlob in hybrid sentiment analysis
    textblob_weight: float = 0.3

    # ==========================================
    # Emotion Parameter Constraints
    # ==========================================

    #: Valid range for valence values (negative to positive)
    valence_range: EmotionRange = (-1.0, 1.0)

    #: Valid range for arousal values (calm to excited)
    arousal_range: EmotionRange = (0.0, 1.0)

    #: Valid range for confidence values (certainty level)
    confidence_range: EmotionRange = (0.0, 1.0)

    # ==========================================
    # SQL Templates Configuration
    # ==========================================

    #: SQL templates for emotion database operations
    sql_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "create_word_emotion_table": SQL_CREATE_WORD_EMOTION_TABLE,
            "create_message_emotion_table": SQL_CREATE_MESSAGE_EMOTION_TABLE,
            "insert_word_emotion": SQL_INSERT_WORD_EMOTION,
            "get_word_emotion": SQL_GET_WORD_EMOTION,
            "insert_message_emotion": SQL_INSERT_MESSAGE_EMOTION,
            "get_message_emotion": SQL_GET_MESSAGE_EMOTION,
        }
    )

    # ==========================================
    # Emotion Classification Configuration
    # ==========================================

    #: Dictionary mapping emotion categories to keywords
    emotion_keywords: EmotionKeywordsDict = field(
        default_factory=lambda: {
            "happiness": ["happy", "joy", "delight", "pleased", "glad", "excited"],
            "sadness": [
                "sad",
                "unhappy",
                "depressed",
                "down",
                "miserable",
                "gloomy",
            ],
            "anger": ["angry", "furious", "enraged", "mad", "irritated", "annoyed"],
            "fear": [
                "afraid",
                "scared",
                "frightened",
                "terrified",
                "anxious",
                "worried",
            ],
            "surprise": [
                "surprised",
                "astonished",
                "amazed",
                "shocked",
                "startled",
            ],
            "disgust": [
                "disgusted",
                "revolted",
                "repulsed",
                "sickened",
                "appalled",
            ],
            "neutral": ["okay", "fine", "neutral", "indifferent", "average"],
        }
    )

    #: Minimum confidence value when no keywords are found
    min_keyword_confidence: float = 0.3

    #: Weight given to keyword matches in emotion classification
    keyword_match_weight: float = 0.6

    # ==========================================
    # Enhanced Configuration Options
    # ==========================================

    #: Registry for dynamic emotion keyword management
    keyword_registry: EmotionKeywordRegistry = field(
        default_factory=EmotionKeywordRegistry
    )

    #: SQL dialect to use for database operations
    sql_dialect: SQLDialect = SQLDialect.SQLITE

    #: Category-specific weights for classification
    per_category_weights: Dict[EmotionCategory, float] = field(default_factory=dict)

    #: Whether to cache emotion analysis results
    enable_caching: bool = True

    #: Time-to-live for cached results in seconds
    cache_ttl: int = 3600

    #: Language code for language-specific processing
    language: str = "en"

    # ==========================================
    # Environment Variable Configuration
    # ==========================================

    #: Mapping of environment variables to configuration attributes
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_ENABLE_VADER": ("enable_vader", bool),
        "WORD_FORGE_VADER_WEIGHT": ("vader_weight", float),
        "WORD_FORGE_TEXTBLOB_WEIGHT": ("textblob_weight", float),
        "WORD_FORGE_MIN_KEYWORD_CONFIDENCE": ("min_keyword_confidence", float),
        "WORD_FORGE_KEYWORD_MATCH_WEIGHT": ("keyword_match_weight", float),
        "WORD_FORGE_ENABLE_CACHING": ("enable_caching", bool),
        "WORD_FORGE_CACHE_TTL": ("cache_ttl", int),
        "WORD_FORGE_LANGUAGE": ("language", str),
    }

    # ==========================================
    # Validation Methods
    # ==========================================

    def is_valid_valence(self, value: float) -> bool:
        """
        Check if a valence value is within the configured range.

        Valence represents the positive or negative quality of an emotion,
        typically ranging from -1.0 (highly negative) to 1.0 (highly positive).

        Args:
            value: The valence value to validate

        Returns:
            bool: True if the value is within the valid range, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_valence(0.7)
            True
            >>> config.is_valid_valence(-1.5)
            False
        """
        min_val, max_val = self.valence_range
        return min_val <= value <= max_val

    def validate_valence(self, value: float) -> EmotionValidationResult:
        """
        Validate a valence value with detailed contextual information.

        Provides a rich validation result containing not just a boolean indicator,
        but also descriptive messages and context about the validation.

        Args:
            value: The valence value to validate

        Returns:
            EmotionValidationResult: Structured validation result with context

        Examples:
            >>> config = EmotionConfig()
            >>> result = config.validate_valence(0.7)
            >>> result.is_valid
            True
            >>> result.message
            'Valence value 0.7 is within valid range (-1.0, 1.0)'

            >>> result = config.validate_valence(-1.5)
            >>> result.is_valid
            False
            >>> result.message
            'Valence value -1.5 is outside valid range (-1.0, 1.0)'
        """
        min_val, max_val = self.valence_range
        is_valid = min_val <= value <= max_val

        if is_valid:
            message = (
                f"Valence value {value} is within valid range ({min_val}, {max_val})"
            )
        else:
            message = (
                f"Valence value {value} is outside valid range ({min_val}, {max_val})"
            )

        return EmotionValidationResult(
            is_valid=is_valid, message=message, value=value, range=self.valence_range
        )

    def is_valid_arousal(self, value: float) -> bool:
        """
        Check if an arousal value is within the configured range.

        Arousal represents the intensity or activation level of an emotion,
        typically ranging from 0.0 (calm) to 1.0 (excited).

        Args:
            value: The arousal value to validate

        Returns:
            bool: True if the value is within the valid range, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_arousal(0.5)
            True
            >>> config.is_valid_arousal(1.2)
            False
        """
        min_val, max_val = self.arousal_range
        return min_val <= value <= max_val

    def validate_arousal(self, value: float) -> EmotionValidationResult:
        """
        Validate an arousal value with detailed contextual information.

        Provides a rich validation result containing not just a boolean indicator,
        but also descriptive messages and context about the validation.

        Args:
            value: The arousal value to validate

        Returns:
            EmotionValidationResult: Structured validation result with context

        Examples:
            >>> config = EmotionConfig()
            >>> result = config.validate_arousal(0.5)
            >>> result.is_valid
            True
            >>> result.message
            'Arousal value 0.5 is within valid range (0.0, 1.0)'

            >>> result = config.validate_arousal(1.2)
            >>> result.is_valid
            False
            >>> result.message
            'Arousal value 1.2 is outside valid range (0.0, 1.0)'
        """
        min_val, max_val = self.arousal_range
        is_valid = min_val <= value <= max_val

        if is_valid:
            message = (
                f"Arousal value {value} is within valid range ({min_val}, {max_val})"
            )
        else:
            message = (
                f"Arousal value {value} is outside valid range ({min_val}, {max_val})"
            )

        return EmotionValidationResult(
            is_valid=is_valid, message=message, value=value, range=self.arousal_range
        )

    def is_valid_confidence(self, value: float) -> bool:
        """
        Check if a confidence value is within the configured range.

        Confidence represents the certainty level in an emotion classification,
        typically ranging from 0.0 (complete uncertainty) to 1.0 (complete certainty).

        Args:
            value: The confidence value to validate

        Returns:
            bool: True if the value is within the valid range, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_confidence(0.9)
            True
            >>> config.is_valid_confidence(-0.1)
            False
        """
        min_val, max_val = self.confidence_range
        return min_val <= value <= max_val

    def validate_confidence(self, value: float) -> EmotionValidationResult:
        """
        Validate a confidence value with detailed contextual information.

        Provides a rich validation result containing not just a boolean indicator,
        but also descriptive messages and context about the validation.

        Args:
            value: The confidence value to validate

        Returns:
            EmotionValidationResult: Structured validation result with context

        Examples:
            >>> config = EmotionConfig()
            >>> result = config.validate_confidence(0.9)
            >>> result.is_valid
            True
            >>> result.message
            'Confidence value 0.9 is within valid range (0.0, 1.0)'

            >>> result = config.validate_confidence(-0.1)
            >>> result.is_valid
            False
            >>> result.message
            'Confidence value -0.1 is outside valid range (0.0, 1.0)'
        """
        min_val, max_val = self.confidence_range
        is_valid = min_val <= value <= max_val

        if is_valid:
            message = (
                f"Confidence value {value} is within valid range ({min_val}, {max_val})"
            )
        else:
            message = f"Confidence value {value} is outside valid range ({min_val}, {max_val})"

        return EmotionValidationResult(
            is_valid=is_valid, message=message, value=value, range=self.confidence_range
        )

    def is_valid_emotion_category_string(self, category: str) -> bool:
        """
        Check if an emotion category string is valid according to configuration.

        Validates that the provided category exists in the configured emotion keywords.

        Args:
            category: The emotion category string to validate

        Returns:
            bool: True if the category exists in configuration, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_emotion_category_string("happiness")
            True
            >>> config.is_valid_emotion_category_string("confusion")  # Not in config
            False
        """
        return category in self.emotion_keywords

    @overload
    def get_keywords_for_emotion(self, emotion: EmotionCategory) -> List[str]: ...

    @overload
    def get_keywords_for_emotion(
        self, emotion: EmotionCategoryLiteral
    ) -> List[str]: ...

    def get_keywords_for_emotion(self, emotion: EmotionCategoryType) -> List[str]:
        """
        Get the list of keywords associated with an emotion category.

        Uses the keyword registry if populated, otherwise falls back to the
        default emotion_keywords dictionary.

        Args:
            emotion: Emotion category (enum or string label)

        Returns:
            List of keywords associated with the emotion

        Raises:
            ValueError: If the emotion category is invalid

        Examples:
            >>> config = EmotionConfig()
            >>> config.get_keywords_for_emotion(EmotionCategory.HAPPINESS)
            ['happy', 'joy', 'delight', 'pleased', 'glad', 'excited']
            >>> config.get_keywords_for_emotion("happiness")  # String label also works
            ['happy', 'joy', 'delight', 'pleased', 'glad', 'excited']
        """
        # Try the registry first
        category = normalize_emotion_category(emotion)
        registry_keywords = self.keyword_registry.get_keywords(category)
        if registry_keywords:
            return registry_keywords

        # Fall back to the default dictionary
        if isinstance(emotion, EmotionCategory):
            emotion_label = emotion.label
        else:
            emotion_label = emotion

        return self.emotion_keywords.get(emotion_label, [])

    # ==========================================
    # Enhanced Configuration Methods
    # ==========================================

    def validate_configuration(self) -> List[str]:
        """
        Validate the entire configuration for consistency and correctness.

        Performs comprehensive checks on all configuration parameters to ensure
        they form a valid, coherent configuration.

        Returns:
            List of validation issues (empty if configuration is valid)

        Examples:
            >>> config = EmotionConfig(vader_weight=0.8, textblob_weight=0.8)
            >>> issues = config.validate_configuration()
            >>> len(issues) > 0  # Configuration has issues
            True
            >>> "weights should sum to 1.0" in issues[0]
            True
        """
        issues: List[str] = []

        # Check weight normalization
        vader_tb_sum = self.vader_weight + self.textblob_weight
        if not isclose(vader_tb_sum, 1.0, abs_tol=0.01):
            issues.append(
                f"Vader and TextBlob weights should sum to 1.0, got {vader_tb_sum}"
            )

        # Check range validations
        for name, range_val in [
            ("valence", self.valence_range),
            ("arousal", self.arousal_range),
            ("confidence", self.confidence_range),
        ]:
            min_val, max_val = range_val
            if min_val >= max_val:
                issues.append(
                    f"{name.capitalize()} range is invalid: {min_val} >= {max_val}"
                )

        # Check per-category weights
        for category in EmotionCategory:
            if category not in self.per_category_weights:
                continue  # Using default weight is fine
            weight = self.per_category_weights[category]
            if not 0.0 <= weight <= 1.0:
                issues.append(
                    f"Invalid weight {weight} for {category.label}, must be between 0.0 and 1.0"
                )

        return issues

    def get_category_weight(self, category: EmotionCategoryType) -> float:
        """
        Get the weight for a specific emotion category, with fallback to default.

        Args:
            category: Emotion category (enum or string label)

        Returns:
            Weight value for the category

        Examples:
            >>> config = EmotionConfig()
            >>> config.per_category_weights[EmotionCategory.HAPPINESS] = 0.9
            >>> config.get_category_weight(EmotionCategory.HAPPINESS)
            0.9
            >>> config.get_category_weight("anger")  # Falls back to default
            0.6
        """
        cat = normalize_emotion_category(category)
        return self.per_category_weights.get(cat, self.keyword_match_weight)

    def get_sql_template(self, operation: str) -> str:
        """
        Get SQL template for a specific operation with dialect support.

        Args:
            operation: Name of the SQL operation to retrieve

        Returns:
            SQL query string adapted to the configured dialect

        Examples:
            >>> config = EmotionConfig(sql_dialect=SQLDialect.POSTGRESQL)
            >>> sql = config.get_sql_template("create_word_emotion_table")
            >>> "SERIAL PRIMARY KEY" in sql  # PostgreSQL syntax
            True
        """
        templates = EmotionSQLTemplates(dialect=self.sql_dialect)
        return templates.get_template(operation)

    def is_valid_emotion_category(self, category: EmotionCategoryType) -> bool:
        """
        Check if an emotion category is valid according to configuration.

        Args:
            category: Emotion category to validate (enum or string label)

        Returns:
            True if the category is valid, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_emotion_category(EmotionCategory.HAPPINESS)
            True
            >>> config.is_valid_emotion_category("happiness")  # String label also works
            True
            >>> config.is_valid_emotion_category("confusion")  # Not in config
            False
        """
        try:
            normalize_emotion_category(category)
            return True
        except ValueError:
            return False

    # ==========================================
    # SQL Template Properties
    # ==========================================
    # Maintained for backward compatibility

    @property
    def SQL_CREATE_WORD_EMOTION_TABLE(self) -> str:
        """
        SQL template for word emotion table creation.

        Returns:
            str: The SQL query for creating the word_emotion table
        """
        return self.sql_templates["create_word_emotion_table"]

    @property
    def SQL_CREATE_MESSAGE_EMOTION_TABLE(self) -> str:
        """
        SQL template for message emotion table creation.

        Returns:
            str: The SQL query for creating the message_emotion table
        """
        return self.sql_templates["create_message_emotion_table"]

    @property
    def SQL_INSERT_WORD_EMOTION(self) -> str:
        """
        SQL template for inserting word emotion data.

        Returns:
            str: The SQL query for inserting data into the word_emotion table
        """
        return self.sql_templates["insert_word_emotion"]

    @property
    def SQL_GET_WORD_EMOTION(self) -> str:
        """
        SQL template for retrieving word emotion data.

        Returns:
            str: The SQL query for retrieving data from the word_emotion table
        """
        return self.sql_templates["get_word_emotion"]

    @property
    def SQL_INSERT_MESSAGE_EMOTION(self) -> str:
        """
        SQL template for inserting message emotion data.

        Returns:
            str: The SQL query for inserting data into the message_emotion table
        """
        return self.sql_templates["insert_message_emotion"]

    @property
    def SQL_GET_MESSAGE_EMOTION(self) -> str:
        """
        SQL template for retrieving message emotion data.

        Returns:
            str: The SQL query for retrieving data from the message_emotion table
        """
        return self.sql_templates["get_message_emotion"]


__all__ = [
    # Main classes
    "EmotionConfig",
    "EmotionKeywordRegistry",
    "EmotionDetectionMetrics",
    # Enums
    "EmotionCategory",
    "SQLDialect",
    # Type definitions
    "EmotionValidationResult",
    "EmotionCategoryLiteral",
    "EmotionKeywordsDict",
    "VaderSentimentScores",
    # Helper functions
    "normalize_emotion_category",
    # SQL Constants (for backward compatibility)
    "SQL_CREATE_WORD_EMOTION_TABLE",
    "SQL_CREATE_MESSAGE_EMOTION_TABLE",
    "SQL_INSERT_WORD_EMOTION",
    "SQL_GET_WORD_EMOTION",
    "SQL_INSERT_MESSAGE_EMOTION",
    "SQL_GET_MESSAGE_EMOTION",
]
