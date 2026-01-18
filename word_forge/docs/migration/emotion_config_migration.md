# Emotion Configuration Migration Guide

> _"Types are contracts, not suggestions; names compress intent without losing meaning."_

This document provides a complete technical pathway for migrating from the legacy string-based emotion configuration to the enhanced enum-based architecture with extended capabilities for runtime optimization, dynamic keyword management, and database portability.

## Core Architecture Changes

```ascii
┌───────────────────────┬───────────────────────┐
│ LEGACY ARCHITECTURE   │ ENHANCED ARCHITECTURE │
├───────────────────────┼───────────────────────┤
│ String Literals       │ Enum-Based Categories │
│ Static Keywords       │ Dynamic Registry      │
│ Single SQL Dialect    │ Dialect Adapters      │
│ Manual Optimization   │ Self-Optimizing       │
│ Basic Validation      │ Comprehensive Checks  │
└───────────────────────┴───────────────────────┘
```

### 1. Type System Enhancements

The emotion system now uses proper enumeration types while maintaining backward compatibility:

```python
# Before: String literals with implicit typing
emotion = "happiness"  # Type information lost at runtime

# After: Type-safe enumerations with rich metadata
emotion = EmotionCategory.HAPPINESS
print(emotion.label)    # "happiness"
print(emotion.weight)   # 0.8
print(emotion.threshold)  # 0.7
```

### 2. Dynamic Keyword Management

Emotion keywords can now be loaded from multiple sources and managed at runtime:

```python
# Before: Static keyword dictionary defined at configuration time
keywords = config.emotion_keywords["happiness"]

# After: Dynamic keyword registry with multiple data sources
config.keyword_registry.load_from_json("domain_specific_emotions.json")
config.keyword_registry.load_from_database(db_connection, "SELECT category, keyword FROM emotion_keywords")
```

### 3. Database Dialect Support

SQL operations now transparently adapt to different database systems:

```python
# Before: SQLite-specific queries only
db.execute(config.SQL_CREATE_WORD_EMOTION_TABLE)

# After: Dialect-aware SQL generation
config.sql_dialect = SQLDialect.POSTGRESQL  # or MYSQL, SQLITE
sql = config.get_sql_template("create_word_emotion_table")
db.execute(sql)  # Contains database-specific syntax
```

### 4. Self-Optimizing Detection

The system can now analyze its own performance and adjust weights accordingly:

```python
# Before: Manual weight tuning based on intuition
config.keyword_match_weight = 0.7  # Arbitrary adjustment

# After: Evidence-based optimization through metrics
metrics = EmotionDetectionMetrics()

# Record ground truth data
for predicted, actual in detection_results:
    metrics.record_detection(predicted, actual)

# Calculate performance metrics
precision = metrics.get_precision(EmotionCategory.FEAR)  # 0.0-1.0
recall = metrics.get_recall(EmotionCategory.ANGER)      # 0.0-1.0
f1 = metrics.get_f1_score(EmotionCategory.HAPPINESS)    # 0.0-1.0

# Apply optimized weights
optimized_weights = metrics.optimize_weights(config)
config.per_category_weights.update(optimized_weights)
```

### 5. Enhanced Validation

Configuration validation is now comprehensive and precise:

```python
# Before: Individual validations without context
if config.is_valid_valence(0.7):
    # Process valid value

# After: Comprehensive validation with detailed diagnostics
issues = config.validate_configuration()
if not issues:
    # Configuration is valid
else:
    for issue in issues:
        logger.warning(f"Configuration issue: {issue}")
```

## Type-Safe Migration Path

### Step 1: Update Imports

```python
# Before
from word_forge.emotion.emotion_config import EmotionConfig

# After
from word_forge.emotion.emotion_config import (
    EmotionConfig,
    EmotionCategory,       # Core enum type
    SQLDialect,            # Database dialect enum
    EmotionKeywordRegistry,  # For dynamic keywords
    EmotionDetectionMetrics, # For performance tracking
    normalize_emotion_category  # For type conversion
)
```

### Step 2: Gradual Type Migration

The enhanced system maintains full backward compatibility while encouraging migration to typed interfaces:

```python
config = EmotionConfig()

# Legacy string-based code continues to work
legacy_keywords = config.get_keywords_for_emotion("happiness")
legacy_is_valid = config.is_valid_emotion_category("happiness")

# New enum-based code provides additional type safety
typed_keywords = config.get_keywords_for_emotion(EmotionCategory.HAPPINESS)
typed_is_valid = config.is_valid_emotion_category(EmotionCategory.HAPPINESS)

# Helper function for migration contexts
try:
    # Convert any emotion representation to enum
    emotion_enum = normalize_emotion_category(emotion_from_database)
except ValueError:
    # Handle invalid emotion categories
    emotion_enum = EmotionCategory.NEUTRAL
```

### Step 3: SQL Dialect Migration

```python
# Migration strategy options:

# Option 1: Continue using direct SQL properties (backward compatible)
db.execute(config.SQL_CREATE_WORD_EMOTION_TABLE)  # Works with SQLite only

# Option 2: Use dialect-aware template retrieval
config.sql_dialect = SQLDialect.POSTGRESQL
create_table_sql = config.get_sql_template("create_word_emotion_table")
db.execute(create_table_sql)  # Now works with PostgreSQL
```

### Step 4: Keyword System Migration

```python
# Migration strategy options:

# Option 1: Continue using static emotion_keywords dictionary
emotion_keywords = config.emotion_keywords["happiness"]

# Option 2: Use the dynamic registry (preferred)
# a) Initialization from existing keywords
for emotion_label, keywords in config.emotion_keywords.items():
    try:
        emotion = EmotionCategory.from_label(emotion_label)
        config.keyword_registry.register_keywords(emotion, keywords)
    except ValueError:
        pass  # Skip invalid emotion categories

# b) Load keywords from external sources
config.keyword_registry.load_from_json("custom_emotions.json")

# c) Add new keywords programmatically
config.keyword_registry.register_keywords(
    EmotionCategory.HAPPINESS,
    ["ecstatic", "thrilled", "delighted"]
)
```

## Advanced Integration Patterns

### Caching Configuration

```python
# Configure caching behavior for performance optimization
config.enable_caching = True
config.cache_ttl = 3600  # seconds

# Access in code
if config.enable_caching:
    cache_key = f"emotion:{word_id}"
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result

    # Compute result if not cached
    result = compute_emotion(word_id)
    cache.set(cache_key, result, ttl=config.cache_ttl)
    return result
```

### Language Support

```python
# Set language for multilingual emotion processing
config.language = "es"  # Spanish

# Access in code
if config.language != "en":
    # Use language-specific processing
    translator = get_translator(config.language)
    translated_text = translator.translate(text)
    # Process using translated text
```

### Category Weight Customization

```python
# Configure custom weights for specific emotion categories
config.per_category_weights[EmotionCategory.HAPPINESS] = 0.9
config.per_category_weights[EmotionCategory.SADNESS] = 0.8

# Retrieve weights with fallback to default
happiness_weight = config.get_category_weight(EmotionCategory.HAPPINESS)  # 0.9
anger_weight = config.get_category_weight(EmotionCategory.ANGER)  # default: keyword_match_weight
```

## Edge Case Handling

### Invalid Emotion Categories

```python
# Before: Direct dictionary access could fail
try:
    keywords = config.emotion_keywords[user_emotion]
except KeyError:
    keywords = []

# After: Validation and normalization
if config.is_valid_emotion_category(user_emotion):
    emotion = normalize_emotion_category(user_emotion)
    keywords = config.get_keywords_for_emotion(emotion)
else:
    keywords = []
```

### Configuration Validation Edge Cases

```python
# Validate configuration before deployment
issues = config.validate_configuration()
if issues:
    # Configuration has problems
    for issue in issues:
        if "weights should sum to 1.0" in issue:
            # Automatically fix weight normalization
            total = config.vader_weight + config.textblob_weight
            config.vader_weight /= total
            config.textblob_weight /= total
        elif "range is invalid" in issue:
            # Log critical issue and abort
            logger.critical(f"Fatal configuration error: {issue}")
            sys.exit(1)
        else:
            # Log warning for other issues
            logger.warning(f"Configuration issue: {issue}")
```

## Migration Testing Strategy

1. **Dual Processing**: Run both systems in parallel and compare outputs

   ```python
   # Process with legacy system
   legacy_result = process_with_legacy(text)

   # Process with enhanced system
   enhanced_result = process_with_enhanced(text)

   # Compare and log differences
   if legacy_result != enhanced_result:
       logger.warning(f"Migration difference: {legacy_result} vs {enhanced_result}")
   ```

2. **Incremental Feature Adoption**: Enable new features one at a time

   ```python
   # Phase 1: Enum-based categories only
   config.use_new_category_system = True

   # Phase 2: Add dynamic keywords
   config.use_keyword_registry = True

   # Phase 3: Add dialect support
   config.sql_dialect = SQLDialect.POSTGRESQL
   ```

3. **Performance Measurement**: Track metrics before and after migration

   ```python
   start_time = time.time()
   result = process_emotions(text)
   processing_time = time.time() - start_time

   logger.info(f"Processing time: {processing_time:.4f}s")
   ```

## Compatibility Matrix

| Feature            | Legacy Support       | Enhanced Capability                    |
| ------------------ | -------------------- | -------------------------------------- |
| Emotion Categories | String literals      | Enum with metadata                     |
| Keyword Management | Static dictionary    | Dynamic registry with multiple sources |
| SQL Support        | SQLite only          | SQLite, PostgreSQL, MySQL              |
| Validation         | Per-value validation | Comprehensive configuration validation |
| Optimization       | Manual tuning        | Self-optimizing through metrics        |
| Caching            | Not available        | Configurable TTL-based caching         |
| Language           | English only         | Configurable language support          |

## Recursive Implementation Example

An example demonstrating multiple features working together:

```python
from word_forge.emotion.emotion_config import (
    EmotionConfig, EmotionCategory, EmotionDetectionMetrics, SQLDialect
)

# 1. Initialize enhanced configuration
config = EmotionConfig()

# 2. Configure for PostgreSQL database
config.sql_dialect = SQLDialect.POSTGRESQL

# 3. Load emotion keywords from multiple sources
config.keyword_registry.load_from_json("domain_keywords.json")
config.keyword_registry.load_from_database(
    db_connection,
    "SELECT emotion, keyword FROM custom_keywords"
)

# 4. Set up metrics collection
metrics = EmotionDetectionMetrics()

# 5. Process text with emotion detection
def analyze_text_emotion(text: str) -> tuple[EmotionCategory, float]:
    # Detect emotion using enhanced system
    emotion, confidence = detect_emotion(text, config)

    # If ground truth is available, record for optimization
    if has_ground_truth(text):
        actual_emotion = get_ground_truth(text)
        metrics.record_detection(emotion, actual_emotion)

        # Periodically optimize weights if enough data
        if metrics.total_detections % 100 == 0:
            optimized_weights = metrics.optimize_weights(config)
            config.per_category_weights.update(optimized_weights)

    return emotion, confidence

# 6. Persist emotion data to database
def save_emotion(message_id: int, emotion: EmotionCategory, confidence: float) -> None:
    # Get dialect-specific SQL
    sql = config.get_sql_template("insert_message_emotion")

    # Execute with appropriate parameters
    db.execute(sql, (message_id, emotion.label, confidence, time.time()))
```

> "Migration isn't just compatibility—it's transforming potential into precision while preserving path integrity."
