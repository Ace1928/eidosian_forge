"""
Demonstration of ParserRefiner functionality.
"""

import os
import time
from pathlib import Path
from typing import Dict, List

from word_forge.parser.parser_refiner import ParserRefiner


def run_demonstration() -> None:
    """
    Run a comprehensive demonstration of the ParserRefiner capabilities.

    This function showcases the full linguistic processing pipeline including:
    - Term extraction and processing
    - Database storage and retrieval
    - Relationship mapping
    - Semantic enrichment
    - Processing statistics
    """
    print("Word Forge Parser Demonstration")
    print("===============================\n")

    # Configure demonstration environment
    data_dir = Path("data").absolute()
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    # Initialize components
    print("\nInitializing parser components...")
    parser = ParserRefiner(data_dir=str(data_dir))
    print("- ParserRefiner initialized")
    print("- Database connection established")
    print("- Queue system ready")

    # Process a set of linguistically rich terms
    demo_terms = ["algorithm", "linguistics", "quantum", "philosophy"]
    results = {}

    print("\nProcessing demonstration terms:")
    for term in demo_terms:
        print(f"\n[PROCESSING] {term}")
        start_time = time.time()
        success = parser.process_word(term)
        processing_time = time.time() - start_time

        results[term] = {"success": success, "time": processing_time}

        print(f"  Status: {'✓ Success' if success else '✗ Failed'}")
        print(f"  Processing time: {processing_time:.2f}s")

        # If successful, retrieve and display the enriched entry
        if success:
            try:
                entry = parser.db_manager.get_word_entry(term)

                # Display core word data
                print(f"\n  [WORD DATA] {term.upper()}")
                print(
                    f"  Definition: {entry['definition'][:100]}..."
                    if len(entry["definition"]) > 100
                    else f"  Definition: {entry['definition']}"
                )
                print(f"  Part of speech: {entry['part_of_speech']}")
                print(
                    f"  Usage examples ({len(entry['usage_examples'])}): "
                    + (
                        entry["usage_examples"][0]
                        if entry["usage_examples"]
                        else "None"
                    )
                )

                # Display relationships
                rel_by_type: Dict[str, List[str]] = {}
                for rel in entry["relationships"]:
                    rel_type = rel["relationship_type"]
                    if rel_type not in rel_by_type:
                        rel_by_type[rel_type] = []
                    rel_by_type[rel_type].append(rel["related_term"])

                print("\n  [RELATIONSHIPS]")
                for rel_type, terms in rel_by_type.items():
                    sample_terms = terms[:5]
                    print(
                        f"  {rel_type.capitalize()} ({len(terms)}): {', '.join(sample_terms)}"
                        + (f" and {len(terms)-5} more..." if len(terms) > 5 else "")
                    )

                # Show queue growth from term
                current_queue_size = parser.queue_manager.size
                print(
                    f"\n  [QUEUE] Discovered {current_queue_size} new candidate terms"
                )

                # Show sample of discovered terms
                if current_queue_size > 0:
                    sample_result = parser.queue_manager.get_sample(5)
                    sample = (
                        sample_result.value if hasattr(sample_result, "value") else []
                    )
                    # Ensure sample is not None before joining
                    sample_list = sample if sample is not None else []
                    print(
                        f"  Sample terms: {', '.join(sample_list)}"
                        + (
                            f" and {current_queue_size-5} more..."
                            if current_queue_size > 5
                            else ""
                        )
                    )

            except Exception as e:
                print(f"  Error retrieving processed data: {str(e)}")

        # Show current statistics after each term
        stats = parser.get_stats()
        print("\n  [STATISTICS]")
        print(f"  Processed: {stats['processed']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Queue size: {stats['queue_size']}")
        print(f"  Unique words: {stats['unique_words']}")

    # Process a term from the queue if available
    print("\n[QUEUE PROCESSING]")
    if parser.queue_manager.size > 0:
        result = parser.queue_manager.dequeue()
        queued_term = result.value if hasattr(result, "value") else str(result)
        if queued_term is not None:
            print(f"Processing term from queue: '{queued_term}'")
            success = parser.process_word(queued_term)
        else:
            print("No valid term retrieved from queue")
            success = False
        print(f"  Status: {'✓ Success' if success else '✗ Failed'}")

        if success:
            print("\n  [SEMANTIC NETWORK GROWTH]")
            print(
                "  This demonstrates how the lexical graph expands through relationship chaining"
            )
            try:
                # Get all words to show how the database has grown
                all_words = parser.db_manager.get_all_words()
                print(f"  Database now contains {len(all_words)} words")
                print("  Sample entries:")
                for i, word in enumerate(all_words[:5]):
                    print(
                        f"   {i+1}. {word['term']}: {word['definition'][:50]}..."
                        if len(word["definition"]) > 50
                        else f"   {i+1}. {word['term']}: {word['definition']}"
                    )
            except Exception as e:
                print(f"  Error retrieving database data: {str(e)}")
    else:
        print("Queue is empty, no additional terms to process")

    # Final statistics
    print("\n[FINAL STATISTICS]")
    final_stats = parser.get_stats()
    for stat, value in final_stats.items():
        print(f"{stat.capitalize()}: {value}")

    # Clean shutdown
    print("\nShutting down parser resources...")
    parser.shutdown()
    print("Demonstration complete")


if __name__ == "__main__":
    run_demonstration()
