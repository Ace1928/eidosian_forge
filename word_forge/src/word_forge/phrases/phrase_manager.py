from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from eidosian_core import eidosian
from word_forge.database.database_manager import DBManager, DatabaseError

LOGGER = logging.getLogger("word_forge.phrases.manager")

@eidosian()
class PhraseManager:
    """Manages phrase inventory, components, and phrase-level templates."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self.db = db_manager or DBManager()

    def upsert_phrase(
        self,
        text: str,
        phoneme_template: Optional[str] = None,
        prosody_priors: Optional[Dict[str, Any]] = None,
        affect_link: Optional[str] = None,
        is_recurring: bool = False,
        novelty_score: float = 1.0,
    ) -> int:
        """Insert or update a phrase record."""
        query = """
        INSERT INTO phrases (text, phoneme_template, prosody_priors, affect_link, is_recurring, novelty_score, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(text)
        DO UPDATE SET
            phoneme_template=excluded.phoneme_template,
            prosody_priors=excluded.prosody_priors,
            affect_link=excluded.affect_link,
            is_recurring=excluded.is_recurring,
            novelty_score=excluded.novelty_score,
            last_seen=excluded.last_seen
        """
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    query,
                    (
                        text.lower().strip(),
                        phoneme_template,
                        json.dumps(prosody_priors) if prosody_priors else None,
                        affect_link,
                        int(is_recurring),
                        novelty_score,
                        time.time(),
                    ),
                )
                row = conn.execute(
                    "SELECT id FROM phrases WHERE text = ?",
                    (text.lower().strip(),),
                ).fetchone()
                if row is None:
                    raise DatabaseError("Failed to retrieve phrase ID after upsert")
                return int(row[0])
        except Exception as e:
            LOGGER.error(f"Failed to upsert phrase '{text}': {e}")
            raise

    def set_phrase_components(self, phrase_id: int, word_ids: List[int]) -> None:
        """Assign word components to a phrase in sequential order."""
        try:
            with self.db.transaction() as conn:
                # Clear existing components
                conn.execute("DELETE FROM phrase_components WHERE phrase_id = ?", (phrase_id,))
                # Insert new components
                for pos, word_id in enumerate(word_ids):
                    conn.execute(
                        "INSERT INTO phrase_components (phrase_id, word_id, position) VALUES (?, ?, ?)",
                        (phrase_id, word_id, pos),
                    )
        except Exception as e:
            LOGGER.error(f"Failed to set components for phrase_id {phrase_id}: {e}")
            raise

    def get_phrase(self, text: str) -> Optional[Dict[str, Any]]:
        """Retrieve full phrase data including component words."""
        query = """
        SELECT p.*, GROUP_CONCAT(w.term, ' ') as component_terms
        FROM phrases p
        JOIN phrase_components pc ON p.id = pc.phrase_id
        JOIN words w ON pc.word_id = w.id
        WHERE p.text = ?
        GROUP BY p.id
        ORDER BY pc.position
        """
        try:
            rows = self.db.execute_query(query, (text.lower().strip(),))
            if not rows:
                return None
            
            data = dict(rows[0])
            if data.get("prosody_priors"):
                data["prosody_priors"] = json.loads(data["prosody_priors"])
            return data
        except Exception as e:
            LOGGER.error(f"Failed to retrieve phrase '{text}': {e}")
            return None

    def track_occurrence(self, text: str, novelty_decay: float = 0.1) -> None:
        """Update last_seen and decrease novelty_score upon phrase occurrence."""
        query = """
        UPDATE phrases
        SET last_seen = ?,
            novelty_score = MAX(0.0, novelty_score - ?)
        WHERE text = ?
        """
        try:
            with self.db.transaction() as conn:
                conn.execute(query, (time.time(), novelty_decay, text.lower().strip()))
        except Exception as e:
            LOGGER.error(f"Failed to track occurrence for phrase '{text}': {e}")
            raise

    def record_realization(
        self,
        phrase_id: int,
        speaker_id: str,
        realization_ipa: Optional[str] = None,
        prosody_actuals: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a specific speaker's realization of a phrase."""
        query = """
        INSERT INTO phrase_realizations (phrase_id, speaker_id, realization_ipa, prosody_actuals, count, last_realized)
        VALUES (?, ?, ?, ?, 1, ?)
        ON CONFLICT(phrase_id, speaker_id)
        DO UPDATE SET
            realization_ipa=COALESCE(excluded.realization_ipa, realization_ipa),
            prosody_actuals=COALESCE(excluded.prosody_actuals, prosody_actuals),
            count=count + 1,
            last_realized=excluded.last_realized
        """
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    query,
                    (
                        phrase_id,
                        speaker_id,
                        realization_ipa,
                        json.dumps(prosody_actuals) if prosody_actuals else None,
                        time.time(),
                    ),
                )
        except Exception as e:
            LOGGER.error(f"Failed to record realization for phrase_id {phrase_id}, speaker {speaker_id}: {e}")
            raise
