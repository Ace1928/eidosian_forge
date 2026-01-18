"""
Demonstration of EmotionManager functionality.
"""

import random

from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_config import EmotionCategory
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.emotion.emotion_types import LLM_AVAILABLE, VADER_AVAILABLE


def main() -> None:
    """Demonstrate EmotionManager functionality with comprehensive emotion analysis."""
    # Initialize with actual DBManager instead of temporary database
    db_manager = DBManager()

    # Create database tables if they don't exist
    db_manager.create_tables()

    # Initialize emotion manager with VADER and LLM if available
    emotion_mgr = EmotionManager(db_manager)

    # Ensure LLM and VADER are enabled if available
    if VADER_AVAILABLE:
        emotion_mgr.config.enable_vader = True
    if LLM_AVAILABLE:
        setattr(emotion_mgr.config, "enable_llm", True)
        setattr(emotion_mgr.config, "llm_weight", 0.6)  # Give significant weight to LLM

    # Re-initialize analysis tools with updated config
    emotion_mgr.init_analysis_tools()

    # Ensure some emotion-related words exist for demonstration
    basic_emotions = [
        (
            "happiness",
            "A state of well-being and contentment",
            "noun",
            ["Finding true happiness is a journey, not a destination."],
        ),
        (
            "sadness",
            "The condition of being sad; sorrow or grief",
            "noun",
            ["She couldn't hide her sadness after the loss."],
        ),
        (
            "anger",
            "A strong feeling of displeasure or hostility",
            "noun",
            ["He managed his anger through meditation techniques."],
        ),
        (
            "fear",
            "An unpleasant emotion caused by anticipation of danger",
            "noun",
            ["Fear of failure kept him from pursuing his dreams."],
        ),
        (
            "surprise",
            "An unexpected event or piece of information",
            "noun",
            ["The birthday party was a complete surprise."],
        ),
    ]

    # Insert basic emotion words if needed
    print("Ensuring emotion vocabulary exists in database...")
    for term, definition, pos, examples in basic_emotions:
        try:
            db_manager.insert_or_update_word(term, definition, pos, examples)
            print(f"  ✓ {term}")
        except Exception as e:
            print(f"  ✗ Could not add '{term}': {e}")

    # Process and track emotions for all words in the database
    try:
        # Get all words from the database
        all_words = db_manager.get_all_words()
        word_count = len(all_words)
        print(f"\nEnriching emotions for all {word_count} words in the database...")

        # First pass: Enrich each word with full emotional data
        enriched_count = 0
        for word_data in all_words:
            word_id = word_data["id"]
            term = word_data["term"]

            # Use the comprehensive enrichment method instead of basic analysis
            emotion_mgr.enrich_word_emotions(term, word_id)

            enriched_count += 1
            if enriched_count % 50 == 0:
                print(f"  Enriched {enriched_count}/{word_count} words...")

        print(f"Completed enriching all {word_count} words.")

        # Second pass: Ensure recursive analysis for all terms
        print(f"\nPerforming recursive analysis on all {word_count} words...")
        recursive_count = 0
        for word_data in all_words:
            term = word_data["term"]

            # Process term recursively to build emotional relationships
            emotion_mgr.analyze_term_recursively(term)

            recursive_count += 1
            if recursive_count % 50 == 0:
                print(f"  Recursively analyzed {recursive_count}/{word_count} words...")

        print(f"Completed recursive analysis for all {word_count} words.")

        # Display a random sample of entries
        sample_size = min(100, word_count)
        sample_indices = random.sample(range(word_count), sample_size)
        sample_words = [all_words[i] for i in sample_indices]

        print(
            f"\nDisplaying random sample of {sample_size} fully enriched word emotions:"
        )
        print("-" * 60)

        for i, word_data in enumerate(sample_words):
            word_id = word_data["id"]
            term = word_data["term"]
            definition = word_data["definition"] or ""

            # Extract usage examples
            usage_examples = []
            if word_data["usage_examples"]:
                usage_str = word_data["usage_examples"]
                usage_examples = usage_str.split("; ") if usage_str else []

            # Retrieve stored emotion data
            stored_emotion = emotion_mgr.get_word_emotion(word_id)
            if not stored_emotion:
                continue

            # Display results
            print(f"Word {i+1}/{sample_size}: '{term}'")
            print(
                f"  - Word Emotion: Valence={stored_emotion['valence']:.2f}, Arousal={stored_emotion['arousal']:.2f}"
            )

            if definition:
                def_valence, def_arousal = emotion_mgr.analyze_text_emotion(definition)
                def_emotion, def_confidence = emotion_mgr.classify_emotion(definition)
                print(f"  - Definition: '{definition}'")
                print(f"    Valence={def_valence:.2f}, Arousal={def_arousal:.2f}")
                print(
                    f"    Classified as: {def_emotion} (confidence: {def_confidence:.2f})"
                )

            # Analyze an example if available
            if usage_examples:
                example = usage_examples[0]
                ex_valence, ex_arousal = emotion_mgr.analyze_text_emotion(example)
                ex_emotion, ex_confidence = emotion_mgr.classify_emotion(example)
                print(f"  - Example: '{example}'")
                print(f"    Valence={ex_valence:.2f}, Arousal={ex_arousal:.2f}")
                print(
                    f"    Classified as: {ex_emotion} (confidence: {ex_confidence:.2f})"
                )

            # Show recursive emotion data if available
            try:
                recursive_data = emotion_mgr.analyze_term_recursively(term)
                print("  - Recursive Analysis:")
                print(
                    f"    Recursive depth: {recursive_data.get('recursive_depth', 'N/A')}"
                )
                meta_emotions = recursive_data.get("meta_emotions", [])
                if meta_emotions and len(meta_emotions) > 0:
                    print(
                        f"    Meta-emotions: {meta_emotions[0].get('label', 'unknown')}, ..."
                    )
                patterns = recursive_data.get("patterns", {})
                if patterns and len(patterns) > 0:
                    pattern_keys = list(patterns.keys())
                    print(
                        f"    Patterns: {pattern_keys[0] if pattern_keys else 'none'}, ..."
                    )
            except Exception:
                pass

            print("-" * 60)

    except Exception as e:
        print(f"Error processing word emotions: {e}")

    # Display analyzer availability
    llm_status = "✓ LLM" if LLM_AVAILABLE else "✗ LLM (not available)"
    vader_status = "✓ VADER" if VADER_AVAILABLE else "✗ VADER (not available)"
    print(f"\nEmotion Analyzers: TextBlob + {vader_status} + {llm_status}")

    # Example: Analyze and track emotions for messages
    messages = [
        (
            100,
            "I'm so happy today! Everything is going GREAT!",
            EmotionCategory.HAPPINESS,
        ),
        (
            101,
            "I feel sad and disappointed about the results.",
            EmotionCategory.SADNESS,
        ),
        (
            102,
            "This product is amazing and exceeds expectations.",
            EmotionCategory.HAPPINESS,
        ),
        (
            103,
            "I'm furious about the terrible customer service.",
            EmotionCategory.ANGER,
        ),
        (104, "Just received the package, it looks fine.", EmotionCategory.NEUTRAL),
        (
            105,
            "OMG this is AWESOME!!! 😊 I love it so much!",
            EmotionCategory.HAPPINESS,
        ),
    ]

    print("\nMessage Emotion Analysis:")
    print("-" * 60)
    for message_id, text, ground_truth in messages:
        # Analyze raw emotional dimensions
        valence, arousal = emotion_mgr.analyze_text_emotion(text)

        # Classify into emotion category with ground truth for metrics
        emotion_data = emotion_mgr.process_message(message_id, text, ground_truth)

        # Retrieve stored data to verify
        stored_emotion = emotion_mgr.get_message_emotion(message_id)

        print(f'Message: "{text}"')
        print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
        print(
            f"Classified as: {emotion_data['emotion_label']} (confidence: {emotion_data['confidence']:.2f})"
        )
        print(f"Stored data: {stored_emotion}")
        print(f"Ground truth: {ground_truth.label}")
        print("-" * 60)

    # Display metrics report
    metrics_report = emotion_mgr.get_metrics_report()
    print("\nEmotion Detection Metrics:")
    print("-" * 60)
    print(f"Total detections: {metrics_report['total_detections']}")
    print(f"Overall accuracy: {metrics_report['overall_accuracy']:.2f}")
    print(f"Average precision: {metrics_report['avg_precision']:.2f}")
    print(f"Average recall: {metrics_report['avg_recall']:.2f}")
    print(f"Average F1 score: {metrics_report['avg_f1']:.2f}")
    print(f"Optimization rounds: {metrics_report['optimization_count']}")

    print("\nCategory-specific metrics:")
    for category, metrics in metrics_report["categories"].items():
        print(f"  {category}:")
        print(f"    Precision: {metrics['precision']:.2f}")
        print(f"    Recall: {metrics['recall']:.2f}")
        print(f"    F1 score: {metrics['f1']:.2f}")
        print(
            f"    TP/FP/FN: {metrics['true_positives']}/{metrics['false_positives']}/{metrics['false_negatives']}"
        )
    print("-" * 60)

    # Demonstrate emotional relationships
    print("\nEmotional Relationship Analysis:")
    print("-" * 60)

    relationship_tests = [
        ("happiness", "joy", "emotional_synonym"),
        ("happiness", "sadness", "emotional_antonym"),
        ("excited", "ecstatic", "intensifies"),
        ("anxiety", "worry", "emotional_component"),
        ("nostalgia", "melancholy", "evokes"),
    ]

    for term1, term2, rel_type in relationship_tests:
        try:
            strength = emotion_mgr.analyze_emotional_relationship(
                term1, term2, rel_type
            )
            print(f"'{term1}' {rel_type} '{term2}': {strength:.2f}")
        except Exception as e:
            print(f"Error analyzing {rel_type} between '{term1}' and '{term2}': {e}")

    print("-" * 60)

    # Test LLM emotional analysis if available
    if emotion_mgr.llm_interface is not None and LLM_AVAILABLE:
        print("\nTesting LLM-enhanced emotional analysis:")
        test_terms = ["serendipity", "melancholy", "exuberant"]
        for term in test_terms:
            emotion_data = emotion_mgr.analyze_term_recursively(term)
            dimensions = emotion_data.get("dimensions", {})
            valence = dimensions.get("valence", 0.0)
            arousal = dimensions.get("arousal", 0.0)
            print(f"  {term}: valence={valence:.2f}, arousal={arousal:.2f}")
    else:
        print("\nLLM integration not available for emotional analysis.")


if __name__ == "__main__":
    main()
