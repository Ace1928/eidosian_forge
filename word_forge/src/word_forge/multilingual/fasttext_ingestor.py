from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from eidosian_core import eidosian

from word_forge.database.database_manager import DBManager
from word_forge.multilingual.multilingual_manager import MultilingualManager
from word_forge.utils.result import Result, failure, success

LOGGER = logging.getLogger("word_forge.multilingual.fasttext")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CandidateTranslation:
    src_lang: str
    src_term: str
    dst_lang: str
    dst_term: str
    score: float
    rank: int


@eidosian()
class FastTextIngestor:
    """Ingest aligned FastText vectors and bootstrap translation candidates."""

    def __init__(self, db_manager: Optional[DBManager] = None, *, storage_path: Optional[Path] = None) -> None:
        self.db = db_manager or DBManager()
        default_storage = Path(self.db.db_path).resolve().parents[2] / "data" / "word_forge_fasttext.sqlite"
        self.storage_path = Path(storage_path or default_storage).resolve()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.storage_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS aligned_vectors (
                    lang TEXT NOT NULL,
                    term TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    norm REAL NOT NULL,
                    source_path TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (lang, term)
                );

                CREATE TABLE IF NOT EXISTS translation_candidates (
                    src_lang TEXT NOT NULL,
                    src_term TEXT NOT NULL,
                    dst_lang TEXT NOT NULL,
                    dst_term TEXT NOT NULL,
                    score REAL NOT NULL,
                    rank_index INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    applied INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (src_lang, src_term, dst_lang, dst_term, method)
                );

                CREATE INDEX IF NOT EXISTS idx_aligned_vectors_lang ON aligned_vectors(lang);
                CREATE INDEX IF NOT EXISTS idx_translation_candidates_src ON translation_candidates(src_lang, src_term);
                CREATE INDEX IF NOT EXISTS idx_translation_candidates_dst ON translation_candidates(dst_lang, dst_term);
                """
            )

    def _iter_vec_rows(self, file_path: str, *, limit: int) -> Iterable[tuple[str, np.ndarray]]:
        parsed = 0
        with open(file_path, "r", encoding="utf-8") as handle:
            first_line = handle.readline().strip().split()
            if len(first_line) != 2 or not all(part.isdigit() for part in first_line):
                handle.seek(0)
            for raw_line in handle:
                if parsed >= limit:
                    break
                parts = raw_line.strip().split()
                if len(parts) < 3:
                    continue
                term = str(parts[0]).strip()
                if not term:
                    continue
                try:
                    vector = np.asarray(parts[1:], dtype=np.float32)
                except Exception:
                    continue
                if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
                    continue
                norm = float(np.linalg.norm(vector))
                if norm <= 0.0:
                    continue
                parsed += 1
                yield term, vector / norm

    def ingest_vectors(self, file_path: str, lang: str, limit: int = 10000) -> Result[int, str]:
        if not os.path.exists(file_path):
            return failure(f"File not found: {file_path}")
        language = str(lang or "").strip().lower()
        if not language:
            return failure("Language code is required")
        try:
            count = 0
            now = _utc_now()
            with self._connect() as conn:
                for term, vector in self._iter_vec_rows(file_path, limit=max(1, int(limit))):
                    conn.execute(
                        """
                        INSERT INTO aligned_vectors (lang, term, dim, vector_json, norm, source_path, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(lang, term) DO UPDATE SET
                            dim=excluded.dim,
                            vector_json=excluded.vector_json,
                            norm=excluded.norm,
                            source_path=excluded.source_path,
                            updated_at=excluded.updated_at
                        """,
                        (
                            language,
                            term,
                            int(vector.size),
                            json.dumps(vector.tolist(), separators=(",", ":")),
                            1.0,
                            str(Path(file_path).resolve()),
                            now,
                        ),
                    )
                    count += 1
            LOGGER.info("Ingested %s aligned vectors for %s", count, language)
            return success(count)
        except Exception as exc:
            LOGGER.error("Failed to ingest vectors from %s: %s", file_path, exc)
            return failure(str(exc))

    def vector_counts(self) -> dict[str, Any]:
        with self._connect() as conn:
            total_vectors = int(conn.execute("SELECT COUNT(*) FROM aligned_vectors").fetchone()[0])
            languages = conn.execute(
                "SELECT lang, COUNT(*) AS count FROM aligned_vectors GROUP BY lang ORDER BY count DESC, lang ASC LIMIT 12"
            ).fetchall()
            candidate_count = int(conn.execute("SELECT COUNT(*) FROM translation_candidates").fetchone()[0])
            applied_count = int(conn.execute("SELECT COUNT(*) FROM translation_candidates WHERE applied = 1").fetchone()[0])
        return {
            "vector_count": total_vectors,
            "language_count": len(languages),
            "top_languages": [{"lang": row["lang"], "count": int(row["count"])} for row in languages],
            "candidate_count": candidate_count,
            "applied_count": applied_count,
            "storage_path": str(self.storage_path),
        }

    def _load_language_vectors(self, lang: str) -> list[tuple[str, np.ndarray]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT term, vector_json FROM aligned_vectors WHERE lang = ? ORDER BY term ASC",
                (str(lang).strip().lower(),),
            ).fetchall()
        out: list[tuple[str, np.ndarray]] = []
        for row in rows:
            try:
                vector = np.asarray(json.loads(row["vector_json"]), dtype=np.float32)
            except Exception:
                continue
            if vector.ndim != 1 or vector.size == 0:
                continue
            out.append((str(row["term"]), vector))
        return out

    def bootstrap_translations(
        self,
        lang_a: str,
        lang_b: str,
        *,
        top_k: int = 1,
        min_score: float = 0.55,
        max_pairs: int = 5000,
        apply: bool = False,
    ) -> Result[dict[str, Any], str]:
        src_lang = str(lang_a or "").strip().lower()
        dst_lang = str(lang_b or "").strip().lower()
        if not src_lang or not dst_lang:
            return failure("Both source and destination languages are required")
        if src_lang == dst_lang:
            return failure("Source and destination languages must differ")

        left = self._load_language_vectors(src_lang)
        right = self._load_language_vectors(dst_lang)
        if not left:
            return failure(f"No aligned vectors ingested for {src_lang}")
        if not right:
            return failure(f"No aligned vectors ingested for {dst_lang}")

        right_terms = [term for term, _vector in right]
        right_matrix = np.stack([vector for _term, vector in right], axis=0)

        candidates: list[CandidateTranslation] = []
        for src_term, src_vector in left:
            scores = right_matrix @ src_vector
            if scores.size == 0:
                continue
            ranked = np.argsort(scores)[::-1][: max(1, int(top_k))]
            for rank, idx in enumerate(ranked, start=1):
                score = float(scores[int(idx)])
                if score < float(min_score):
                    continue
                candidates.append(
                    CandidateTranslation(
                        src_lang=src_lang,
                        src_term=src_term,
                        dst_lang=dst_lang,
                        dst_term=right_terms[int(idx)],
                        score=round(score, 6),
                        rank=rank,
                    )
                )
                if len(candidates) >= max(1, int(max_pairs)):
                    break
            if len(candidates) >= max(1, int(max_pairs)):
                break

        applied = 0
        manager = MultilingualManager(self.db)
        now = _utc_now()
        with self._connect() as conn:
            for row in candidates:
                conn.execute(
                    """
                    INSERT INTO translation_candidates
                    (src_lang, src_term, dst_lang, dst_term, score, rank_index, method, applied, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(src_lang, src_term, dst_lang, dst_term, method) DO UPDATE SET
                        score=excluded.score,
                        rank_index=excluded.rank_index,
                        applied=excluded.applied,
                        updated_at=excluded.updated_at
                    """,
                    (
                        row.src_lang,
                        row.src_term,
                        row.dst_lang,
                        row.dst_term,
                        float(row.score),
                        int(row.rank),
                        "fasttext_aligned",
                        1 if apply else 0,
                        now,
                    ),
                )

        if apply:
            for row in candidates:
                lexeme_id = manager.upsert_lexeme(lemma=row.src_term, lang=row.src_lang, source="fasttext_aligned")
                self.db.add_translation(
                    lexeme_id=lexeme_id,
                    target_lang=row.dst_lang,
                    target_term=row.dst_term,
                    relation="fasttext_bootstrap",
                    source="fasttext_aligned",
                )
                if row.dst_lang == "en":
                    self.db.update_lexeme_base(lemma=row.src_term, lang=row.src_lang, base_term=row.dst_term)
                    manager.ensure_base_word(row.dst_term, definition=f"FastText aligned base term for {row.src_term}")
                applied += 1

        return success(
            {
                "src_lang": src_lang,
                "dst_lang": dst_lang,
                "candidate_count": len(candidates),
                "applied_count": applied,
                "top_candidates": [
                    {
                        "src_term": row.src_term,
                        "dst_term": row.dst_term,
                        "score": row.score,
                        "rank": row.rank,
                    }
                    for row in candidates[:20]
                ],
            }
        )
