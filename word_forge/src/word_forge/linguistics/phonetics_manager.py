from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Union

from eidosian_core import eidosian
from word_forge.database.database_manager import DBManager, DatabaseError

LOGGER = logging.getLogger("word_forge.linguistics.phonetics")

@eidosian()
class PhoneticsManager:
    """Manages phonetic data, IPA representations, and pronunciation variants."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        """Initialize the phonetics manager with a database manager."""
        self.db = db_manager or DBManager()

    def upsert_phonetics(
        self,
        word_id: int,
        ipa: str,
        arpabet: Optional[str] = None,
        stress_pattern: Optional[str] = None,
        source: str = "manual",
        is_exception: bool = False,
    ) -> int:
        """Insert or update phonetic information for a word."""
        query = """
        INSERT INTO phonetics (word_id, ipa, arpabet, stress_pattern, source, is_exception, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(word_id, ipa)
        DO UPDATE SET
            arpabet=excluded.arpabet,
            stress_pattern=excluded.stress_pattern,
            source=excluded.source,
            is_exception=excluded.is_exception,
            last_updated=excluded.last_updated
        """
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    query,
                    (word_id, ipa, arpabet, stress_pattern, source, int(is_exception), time.time()),
                )
                # Fetch the ID of the inserted/updated record
                row = conn.execute(
                    "SELECT id FROM phonetics WHERE word_id = ? AND ipa = ?",
                    (word_id, ipa),
                ).fetchone()
                if row is None:
                    raise DatabaseError("Failed to retrieve phonetic ID after upsert")
                return int(row[0])
        except Exception as e:
            LOGGER.error(f"Failed to upsert phonetics for word_id {word_id}: {e}")
            raise

    def add_variant(
        self,
        phonetic_id: int,
        variant_ipa: str,
        context: Optional[str] = None,
        frequency_weight: float = 1.0,
    ) -> None:
        """Add a pronunciation variant for an existing phonetic record."""
        query = """
        INSERT INTO pronunciation_variants (phonetic_id, variant_ipa, context, frequency_weight)
        VALUES (?, ?, ?, ?)
        """
        try:
            with self.db.transaction() as conn:
                conn.execute(query, (phonetic_id, variant_ipa, context, frequency_weight))
        except Exception as e:
            LOGGER.error(f"Failed to add pronunciation variant for phonetic_id {phonetic_id}: {e}")
            raise

    def get_phonetics(self, word_id: int) -> List[Dict[str, Any]]:
        """Retrieve all phonetic records and their variants for a word."""
        query = """
        SELECT p.*, GROUP_CONCAT(v.variant_ipa || '|' || COALESCE(v.context, '') || '|' || v.frequency_weight, ';') as variants
        FROM phonetics p
        LEFT JOIN pronunciation_variants v ON p.id = v.phonetic_id
        WHERE p.word_id = ?
        GROUP BY p.id
        """
        try:
            rows = self.db.execute_query(query, (word_id,))
            results = []
            for row in rows:
                data = dict(row)
                variants = []
                if data.get("variants"):
                    for v_str in data["variants"].split(";"):
                        v_parts = v_str.split("|")
                        if len(v_parts) == 3:
                            variants.append({
                                "ipa": v_parts[0],
                                "context": v_parts[1],
                                "weight": float(v_parts[2])
                            })
                data["variants"] = variants
                results.append(data)
            return results
        except Exception as e:
            LOGGER.error(f"Failed to retrieve phonetics for word_id {word_id}: {e}")
            return []

    def get_phonetics_by_term(self, term: str) -> List[Dict[str, Any]]:
        """Retrieve phonetics for a word term."""
        try:
            word_id = self.db.get_word_id(term)
            return self.get_phonetics(word_id)
        except Exception:
            return []
