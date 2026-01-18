"""
Demonstration of DBManager functionality.
"""

from word_forge.database.database_manager import DatabaseError, DBManager


def main() -> None:
    """
    Demonstrate DBManager functionality with basic CRUD operations.

    This function provides a simple demonstration of key database operations:
    - Creating the database schema
    - Inserting and updating word entries
    - Creating relationships between words
    - Retrieving word entries with their relationships

    Examples:
        >>> # Run the demonstration
        >>> main()
    """
    # Initialize with a test database
    db_path = "test_database.sqlite"
    db_manager = DBManager(db_path=db_path)

    try:
        # Create database tables
        print("Creating database schema...")
        db_manager.create_tables()

        # Insert some sample words
        print("\nInserting sample words...")
        db_manager.insert_or_update_word(
            "algorithm",
            "A step-by-step procedure for calculations or problem-solving.",
            "noun",
            [
                "The sorting algorithm runs in O(n log n) time.",
                "She developed a new algorithm for image recognition.",
            ],
        )

        db_manager.insert_or_update_word(
            "recursion",
            "A process where the solution depends on solutions to smaller instances of the same problem.",
            "noun",
            [
                "Recursion is often used in algorithms that work with tree structures.",
                "Understanding recursion requires understanding recursion.",
            ],
        )

        # Create relationships
        print("\nCreating relationships between words...")
        db_manager.insert_relationship("algorithm", "recursion", "related_concept")

        # Retrieve and display word entries
        print("\nRetrieving complete word entries:")
        algorithm_entry = db_manager.get_word_entry("algorithm")
        print(f"\nTerm: {algorithm_entry['term']}")
        print(f"Definition: {algorithm_entry['definition']}")
        print(f"Part of speech: {algorithm_entry['part_of_speech']}")
        print("Usage examples:")
        for example in algorithm_entry["usage_examples"]:
            print(f"  - {example}")
        print("Relationships:")
        for rel in algorithm_entry["relationships"]:
            print(f"  - {rel['relationship_type']} to {rel['related_term']}")

        # Clean up
        print("\nClosing database connection...")
        db_manager.close()
        print("Demonstration completed successfully!")

    except DatabaseError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
