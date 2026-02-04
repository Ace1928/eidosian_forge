#!/usr/bin/env python3
"""
Seed Multidimensional Graph Data.

This script populates the Word Forge database with rich example data covering:
- Lexical: Synonyms, Antonyms, Hypernyms
- Emotional: Valence, Arousal, Evocations
- Semantic: Is-a, Part-of relationships
"""

import sys
import time
from pathlib import Path

# Ensure we can import word_forge
sys.path.insert(0, str(Path(__file__).parents[1].resolve() / "src"))
sys.path.insert(0, str(Path(__file__).parents[2].resolve() / "lib"))

from word_forge.database.database_manager import DBManager

def seed_data():
    db_manager = DBManager()
    db_manager.create_tables()
    
    # --- 1. Core Terms (Lexical) ---
    print("Seeding Lexical Terms...")
    terms = [
        ("joy", "A feeling of great pleasure and happiness.", "noun", "She felt a surge of joy."),
        ("sadness", "The condition or quality of being sad.", "noun", "A deep sadness overcame him."),
        ("fury", "Wild or violent anger.", "noun", "His eyes blazed with fury."),
        ("serenity", "The state of being calm, peaceful, and untroubled.", "noun", "The serenity of the lake at dawn."),
        ("storm", "A violent disturbance of the atmosphere.", "noun", "The storm raged all night."),
        ("ocean", "A very large expanse of sea.", "noun", "We sailed across the ocean."),
    ]
    
    for term, definition, pos, example in terms:
        db_manager.insert_or_update_word(
            term=term,
            definition=definition,
            part_of_speech=pos,
            usage_examples=[example]
        )

    # --- 2. Relationships (Multidimensional) ---
    print("Seeding Relationships...")
    relationships = [
        # Lexical
        ("joy", "happiness", "synonym"),
        ("sadness", "happiness", "antonym"),
        ("fury", "anger", "synonym"),
        
        # Semantic (Conceptual)
        ("storm", "weather", "is_a"),
        ("ocean", "water", "made_of"),
        
        # Emotional / Affective
        ("joy", "serenity", "related_emotion"),
        ("storm", "fear", "evokes"),
        ("ocean", "serenity", "evokes"),
        ("fury", "violence", "leads_to"),
    ]

    for u, v, rel_type in relationships:
        # Ensure target words exist (simplified)
        db_manager.insert_or_update_word(term=v)
        try:
            db_manager.insert_relationship(u, v, rel_type)
        except Exception as e:
            print(f"Skipping existing: {u} -> {v} ({e})")

    # --- 3. Emotional Metadata (Valence/Arousal) ---
    print("Seeding Emotional Attributes...")
    
    # Check if word_emotion table exists, otherwise create/use it
    with db_manager.transaction() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS word_emotion (
                word_id INTEGER PRIMARY KEY,
                valence REAL,
                arousal REAL,
                dominance REAL,
                label TEXT,
                FOREIGN KEY(word_id) REFERENCES words(id)
            )
        """)
    
    emotions = {
        "joy": (0.9, 0.8, "positive"),
        "sadness": (-0.8, -0.4, "negative"),
        "fury": (-0.7, 0.9, "high_arousal"),
        "serenity": (0.8, 0.1, "low_arousal"),
        "storm": (-0.2, 0.7, "intense"),
    }
    
    for term, (val, aro, label) in emotions.items():
        try:
            # We must not use execute_scalar inside a manual transaction block if execute_scalar
            # opens its own connection/context unless it's designed to nest. 
            # get_word_id uses execute_scalar which uses get_connection.
            # get_connection yields a connection. 
            
            # Safest is to get ID first, then transaction.
            wid = db_manager.get_word_id(term)
            
            with db_manager.transaction() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO word_emotion (word_id, valence, arousal, label) VALUES (?, ?, ?, ?)",
                    (wid, val, aro, label)
                )
        except Exception as e:
            print(f"Error seeding emotion for {term}: {e}")

    print("✅ Seeding Complete.")
    db_manager.close()

if __name__ == "__main__":
    seed_data()
