"""
Emotional Configuration module for Word Forge.

This module provides configuration parameters for emotion analysis operations,
including sentiment analysis settings, emotional classification rules, and
database schema definitions for storing emotion-related data.

The module contains:
    - SQL template constants for database operations
    - EmotionConfig dataclass with configurable parameters
    - Validation methods for emotional measurements
    - Property accessors for backward compatibility
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Final, List

from word_forge.configs.config_essentials import EmotionRange
from word_forge.configs.config_types import EnvMapping

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
        ENV_VARS: Mapping of environment variables to config attributes

    Examples:
        >>> config = EmotionConfig()
        >>> config.is_valid_valence(0.5)
        True
        >>> config.is_valid_valence(2.0)  # Out of range
        False
        >>> config.SQL_GET_WORD_EMOTION  # Access SQL via property
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
    emotion_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "happiness": ["happy", "joy", "delight", "pleased", "glad", "excited"],
            "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "gloomy"],
            "anger": ["angry", "furious", "enraged", "mad", "irritated", "annoyed"],
            "fear": [
                "afraid",
                "scared",
                "frightened",
                "terrified",
                "anxious",
                "worried",
            ],
            "surprise": ["surprised", "astonished", "amazed", "shocked", "startled"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"],
            "neutral": ["okay", "fine", "neutral", "indifferent", "average"],
        }
    )

    #: Minimum confidence value when no keywords are found
    min_keyword_confidence: float = 0.3

    #: Weight given to keyword matches in emotion classification
    keyword_match_weight: float = 0.6

    # ==========================================
    # Environment Variable Configuration
    # ==========================================

    #: Mapping of environment variables to configuration attributes
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_ENABLE_VADER": ("enable_vader", bool),
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
