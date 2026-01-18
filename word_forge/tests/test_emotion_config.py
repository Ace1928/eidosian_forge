"""Tests for word_forge.emotion.emotion_config module.

This module tests the emotion configuration classes including EmotionCategory,
EmotionConfig, EmotionDetectionMetrics, EmotionKeywordRegistry, and related
functionality.
"""

import pytest

from word_forge.emotion.emotion_config import (
    EmotionCategory,
    EmotionConfig,
    EmotionDetectionMetrics,
    EmotionKeywordRegistry,
    EmotionSQLTemplates,
    EmotionValidationResult,
    SQLDialect,
    normalize_emotion_category,
)


class TestEmotionCategory:
    """Tests for the EmotionCategory enum."""

    def test_all_categories_exist(self):
        """Test that all expected categories exist."""
        assert EmotionCategory.HAPPINESS
        assert EmotionCategory.SADNESS
        assert EmotionCategory.ANGER
        assert EmotionCategory.FEAR
        assert EmotionCategory.SURPRISE
        assert EmotionCategory.DISGUST
        assert EmotionCategory.NEUTRAL

    def test_category_count(self):
        """Test the number of categories."""
        assert len(EmotionCategory) == 7

    def test_category_has_label(self):
        """Test that categories have labels."""
        assert EmotionCategory.HAPPINESS.label == "happiness"
        assert EmotionCategory.SADNESS.label == "sadness"

    def test_category_has_weight(self):
        """Test that categories have weights."""
        assert 0.0 <= EmotionCategory.HAPPINESS.weight <= 1.0
        assert 0.0 <= EmotionCategory.SADNESS.weight <= 1.0

    def test_category_has_threshold(self):
        """Test that categories have thresholds."""
        assert 0.0 <= EmotionCategory.HAPPINESS.threshold <= 1.0
        assert 0.0 <= EmotionCategory.SADNESS.threshold <= 1.0

    def test_from_label_valid(self):
        """Test from_label with valid labels."""
        assert EmotionCategory.from_label("happiness") == EmotionCategory.HAPPINESS
        assert EmotionCategory.from_label("sadness") == EmotionCategory.SADNESS
        assert EmotionCategory.from_label("anger") == EmotionCategory.ANGER

    def test_from_label_invalid(self):
        """Test from_label with invalid label."""
        with pytest.raises(ValueError, match="Unknown emotion category"):
            EmotionCategory.from_label("invalid")

    def test_str_representation(self):
        """Test string representation."""
        assert str(EmotionCategory.HAPPINESS) == "happiness"
        assert str(EmotionCategory.SADNESS) == "sadness"


class TestNormalizeEmotionCategory:
    """Tests for the normalize_emotion_category function."""

    def test_normalize_enum(self):
        """Test normalizing an enum value."""
        result = normalize_emotion_category(EmotionCategory.HAPPINESS)
        assert result == EmotionCategory.HAPPINESS

    def test_normalize_string(self):
        """Test normalizing a string label."""
        result = normalize_emotion_category("happiness")
        assert result == EmotionCategory.HAPPINESS

    def test_normalize_invalid_string(self):
        """Test normalizing an invalid string."""
        with pytest.raises(ValueError, match="Invalid emotion category"):
            normalize_emotion_category("invalid")


class TestEmotionValidationResult:
    """Tests for the EmotionValidationResult named tuple."""

    def test_create_valid_result(self):
        """Test creating a valid result."""
        result = EmotionValidationResult(
            is_valid=True,
            message="Value is valid",
            value=0.5,
            range=(-1.0, 1.0),
        )
        assert result.is_valid is True
        assert result.message == "Value is valid"
        assert result.value == 0.5
        assert result.range == (-1.0, 1.0)

    def test_create_invalid_result(self):
        """Test creating an invalid result."""
        result = EmotionValidationResult(
            is_valid=False,
            message="Value out of range",
            value=2.0,
            range=(-1.0, 1.0),
        )
        assert result.is_valid is False


class TestSQLDialect:
    """Tests for the SQLDialect enum."""

    def test_dialects_exist(self):
        """Test that all dialects exist."""
        assert SQLDialect.SQLITE
        assert SQLDialect.POSTGRESQL
        assert SQLDialect.MYSQL

    def test_dialect_count(self):
        """Test the number of dialects."""
        assert len(SQLDialect) == 3


class TestEmotionSQLTemplates:
    """Tests for the EmotionSQLTemplates class."""

    def test_default_dialect(self):
        """Test default dialect is SQLite."""
        templates = EmotionSQLTemplates()
        assert templates.dialect == SQLDialect.SQLITE

    def test_get_template_sqlite(self):
        """Test getting template with SQLite dialect."""
        templates = EmotionSQLTemplates(dialect=SQLDialect.SQLITE)
        sql = templates.get_template("create_word_emotion_table")
        assert "INTEGER PRIMARY KEY" in sql
        assert "CREATE TABLE IF NOT EXISTS" in sql

    def test_get_template_postgresql(self):
        """Test getting template with PostgreSQL dialect."""
        templates = EmotionSQLTemplates(dialect=SQLDialect.POSTGRESQL)
        sql = templates.get_template("create_word_emotion_table")
        assert "SERIAL PRIMARY KEY" in sql

    def test_get_template_mysql(self):
        """Test getting template with MySQL dialect."""
        templates = EmotionSQLTemplates(dialect=SQLDialect.MYSQL)
        sql = templates.get_template("create_word_emotion_table")
        assert "AUTO_INCREMENT" in sql

    def test_get_invalid_template(self):
        """Test getting invalid template returns empty string."""
        templates = EmotionSQLTemplates()
        sql = templates.get_template("invalid_operation")
        assert sql == ""


class TestEmotionKeywordRegistry:
    """Tests for the EmotionKeywordRegistry class."""

    def test_empty_registry(self):
        """Test empty registry."""
        registry = EmotionKeywordRegistry()
        assert registry.get_keywords(EmotionCategory.HAPPINESS) == []

    def test_register_keywords(self):
        """Test registering keywords."""
        registry = EmotionKeywordRegistry()
        registry.register_keywords(EmotionCategory.HAPPINESS, ["joyful", "elated"])
        keywords = registry.get_keywords(EmotionCategory.HAPPINESS)
        assert "joyful" in keywords
        assert "elated" in keywords

    def test_register_keywords_with_string(self):
        """Test registering keywords with string label."""
        registry = EmotionKeywordRegistry()
        registry.register_keywords("sadness", ["miserable", "depressed"])
        keywords = registry.get_keywords("sadness")
        assert "miserable" in keywords
        assert "depressed" in keywords

    def test_register_keywords_extends(self):
        """Test that registering keywords extends existing list."""
        registry = EmotionKeywordRegistry()
        registry.register_keywords(EmotionCategory.HAPPINESS, ["joyful"])
        registry.register_keywords(EmotionCategory.HAPPINESS, ["elated"])
        keywords = registry.get_keywords(EmotionCategory.HAPPINESS)
        assert "joyful" in keywords
        assert "elated" in keywords

    def test_register_keywords_no_duplicates(self):
        """Test that duplicate keywords are not added."""
        registry = EmotionKeywordRegistry()
        registry.register_keywords(EmotionCategory.HAPPINESS, ["joyful"])
        registry.register_keywords(EmotionCategory.HAPPINESS, ["joyful", "elated"])
        keywords = registry.get_keywords(EmotionCategory.HAPPINESS)
        assert keywords.count("joyful") == 1

    def test_clear(self):
        """Test clearing the registry."""
        registry = EmotionKeywordRegistry()
        registry.register_keywords(EmotionCategory.HAPPINESS, ["joyful"])
        registry.clear()
        assert registry.get_keywords(EmotionCategory.HAPPINESS) == []
        assert registry.get_sources() == []


class TestEmotionDetectionMetrics:
    """Tests for the EmotionDetectionMetrics class."""

    def test_initial_values(self):
        """Test initial metric values."""
        metrics = EmotionDetectionMetrics()
        assert metrics.total_detections == 0

    def test_record_true_positive(self):
        """Test recording a true positive."""
        metrics = EmotionDetectionMetrics()
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
        assert metrics.total_detections == 1
        assert metrics.true_positives[EmotionCategory.HAPPINESS] == 1

    def test_record_false_positive(self):
        """Test recording a false positive."""
        metrics = EmotionDetectionMetrics()
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.SADNESS)
        assert metrics.total_detections == 1
        assert metrics.false_positives[EmotionCategory.HAPPINESS] == 1
        assert metrics.false_negatives[EmotionCategory.SADNESS] == 1

    def test_record_with_strings(self):
        """Test recording with string labels."""
        metrics = EmotionDetectionMetrics()
        metrics.record_detection("happiness", "happiness")
        assert metrics.total_detections == 1
        assert metrics.true_positives[EmotionCategory.HAPPINESS] == 1

    def test_get_precision(self):
        """Test precision calculation."""
        metrics = EmotionDetectionMetrics()
        # 2 true positives, 1 false positive = 2/3 precision
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.SADNESS)
        precision = metrics.get_precision(EmotionCategory.HAPPINESS)
        assert abs(precision - (2 / 3)) < 0.001

    def test_get_precision_zero(self):
        """Test precision when no predictions."""
        metrics = EmotionDetectionMetrics()
        precision = metrics.get_precision(EmotionCategory.HAPPINESS)
        assert precision == 0.0

    def test_get_recall(self):
        """Test recall calculation."""
        metrics = EmotionDetectionMetrics()
        # 2 true positives, 1 false negative = 2/3 recall
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
        metrics.record_detection(EmotionCategory.SADNESS, EmotionCategory.HAPPINESS)
        recall = metrics.get_recall(EmotionCategory.HAPPINESS)
        assert abs(recall - (2 / 3)) < 0.001

    def test_get_recall_zero(self):
        """Test recall when no true positives or false negatives."""
        metrics = EmotionDetectionMetrics()
        recall = metrics.get_recall(EmotionCategory.HAPPINESS)
        assert recall == 0.0

    def test_get_f1_score(self):
        """Test F1 score calculation."""
        metrics = EmotionDetectionMetrics()
        # Perfect precision and recall = F1 of 1.0
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
        f1 = metrics.get_f1_score(EmotionCategory.HAPPINESS)
        assert f1 == 1.0

    def test_get_f1_score_zero(self):
        """Test F1 score when both precision and recall are 0."""
        metrics = EmotionDetectionMetrics()
        f1 = metrics.get_f1_score(EmotionCategory.HAPPINESS)
        assert f1 == 0.0

    def test_reset(self):
        """Test resetting metrics."""
        metrics = EmotionDetectionMetrics()
        metrics.record_detection(EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS)
        metrics.reset()
        assert metrics.total_detections == 0
        assert metrics.true_positives[EmotionCategory.HAPPINESS] == 0


class TestEmotionConfig:
    """Tests for the EmotionConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmotionConfig()
        assert config.enable_vader is True
        assert config.vader_weight == 0.7
        assert config.textblob_weight == 0.3
        assert config.valence_range == (-1.0, 1.0)
        assert config.arousal_range == (0.0, 1.0)
        assert config.confidence_range == (0.0, 1.0)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmotionConfig(
            enable_vader=False,
            vader_weight=0.5,
            textblob_weight=0.5,
        )
        assert config.enable_vader is False
        assert config.vader_weight == 0.5
        assert config.textblob_weight == 0.5

    def test_emotion_keywords_default(self):
        """Test default emotion keywords."""
        config = EmotionConfig()
        assert "happy" in config.emotion_keywords["happiness"]
        assert "sad" in config.emotion_keywords["sadness"]
        assert "angry" in config.emotion_keywords["anger"]

    def test_sql_templates_default(self):
        """Test default SQL templates."""
        config = EmotionConfig()
        assert "create_word_emotion_table" in config.sql_templates
        assert "create_message_emotion_table" in config.sql_templates
        assert "insert_word_emotion" in config.sql_templates
        assert "get_word_emotion" in config.sql_templates

    def test_keyword_registry(self):
        """Test keyword registry is initialized."""
        config = EmotionConfig()
        assert isinstance(config.keyword_registry, EmotionKeywordRegistry)

    def test_sql_dialect_default(self):
        """Test default SQL dialect."""
        config = EmotionConfig()
        assert config.sql_dialect == SQLDialect.SQLITE

    def test_caching_default(self):
        """Test default caching configuration."""
        config = EmotionConfig()
        assert config.enable_caching is True
        assert config.cache_ttl == 3600

    def test_language_default(self):
        """Test default language."""
        config = EmotionConfig()
        assert config.language == "en"


class TestEmotionConfigValidation:
    """Tests for EmotionConfig validation methods."""

    def test_is_valid_valence_valid(self):
        """Test valid valence values."""
        config = EmotionConfig()
        assert config.is_valid_valence(0.0) is True
        assert config.is_valid_valence(-1.0) is True
        assert config.is_valid_valence(1.0) is True
        assert config.is_valid_valence(0.5) is True

    def test_is_valid_valence_invalid(self):
        """Test invalid valence values."""
        config = EmotionConfig()
        assert config.is_valid_valence(1.5) is False
        assert config.is_valid_valence(-1.5) is False

    def test_is_valid_arousal_valid(self):
        """Test valid arousal values."""
        config = EmotionConfig()
        assert config.is_valid_arousal(0.0) is True
        assert config.is_valid_arousal(1.0) is True
        assert config.is_valid_arousal(0.5) is True

    def test_is_valid_arousal_invalid(self):
        """Test invalid arousal values."""
        config = EmotionConfig()
        assert config.is_valid_arousal(-0.1) is False
        assert config.is_valid_arousal(1.5) is False

    def test_validate_valence(self):
        """Test validate_valence returns detailed result."""
        config = EmotionConfig()
        result = config.validate_valence(0.5)
        assert result.is_valid is True
        assert result.value == 0.5
        assert result.range == (-1.0, 1.0)

    def test_validate_valence_invalid(self):
        """Test validate_valence with invalid value."""
        config = EmotionConfig()
        result = config.validate_valence(2.0)
        assert result.is_valid is False
        assert "outside valid range" in result.message

    def test_validate_arousal(self):
        """Test validate_arousal returns detailed result."""
        config = EmotionConfig()
        result = config.validate_arousal(0.5)
        assert result.is_valid is True

    def test_validate_arousal_invalid(self):
        """Test validate_arousal with invalid value."""
        config = EmotionConfig()
        result = config.validate_arousal(-0.5)
        assert result.is_valid is False


class TestEmotionConfigSQLMethods:
    """Tests for EmotionConfig SQL methods."""

    def test_get_sql_template(self):
        """Test get_sql_template method."""
        config = EmotionConfig()
        sql = config.get_sql_template("create_word_emotion_table")
        assert "CREATE TABLE IF NOT EXISTS" in sql
        assert "word_emotion" in sql

    def test_get_sql_template_invalid(self):
        """Test get_sql_template with invalid operation."""
        config = EmotionConfig()
        sql = config.get_sql_template("invalid_operation")
        assert sql == ""


class TestEmotionConfigKeywordMethods:
    """Tests for EmotionConfig keyword methods."""

    def test_get_keywords_for_emotion(self):
        """Test get_keywords_for_emotion method."""
        config = EmotionConfig()
        keywords = config.get_keywords_for_emotion(EmotionCategory.HAPPINESS)
        assert "happy" in keywords

    def test_get_keywords_for_emotion_string(self):
        """Test get_keywords_for_emotion with string label."""
        config = EmotionConfig()
        keywords = config.get_keywords_for_emotion(EmotionCategory.SADNESS)
        assert "sad" in keywords

    def test_get_category_weight(self):
        """Test get_category_weight method."""
        config = EmotionConfig()
        weight = config.get_category_weight(EmotionCategory.HAPPINESS)
        assert 0.0 <= weight <= 1.0


class TestEmotionDetectionMetricsOptimization:
    """Tests for EmotionDetectionMetrics optimization."""

    def test_optimize_weights_insufficient_data(self):
        """Test optimization with insufficient data."""
        metrics = EmotionDetectionMetrics()
        config = EmotionConfig()
        # With less than 10 detections, returns current weights
        weights = metrics.optimize_weights(config)
        assert len(weights) == len(EmotionCategory)

    def test_optimize_weights_with_data(self):
        """Test optimization with sufficient data."""
        metrics = EmotionDetectionMetrics()
        config = EmotionConfig()
        # Record 15 perfect predictions
        for _ in range(15):
            metrics.record_detection(
                EmotionCategory.HAPPINESS, EmotionCategory.HAPPINESS
            )
        weights = metrics.optimize_weights(config)
        # High precision should increase weight
        assert EmotionCategory.HAPPINESS in weights
