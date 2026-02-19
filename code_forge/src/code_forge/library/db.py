from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from code_forge.library.similarity import hamming_distance64, tokenize_code_text, token_jaccard

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass(frozen=True)
class CodeUnit:
    unit_type: str
    name: str
    file_path: str
    language: str = "python"
    qualified_name: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    col_start: Optional[int] = None
    col_end: Optional[int] = None
    content_hash: Optional[str] = None
    parent_id: Optional[str] = None
    run_id: Optional[str] = None
    complexity: Optional[float] = None
    normalized_hash: Optional[str] = None
    simhash64: Optional[str] = None  # 16-char hex
    token_count: Optional[int] = None
    semantic_text: Optional[str] = None


class CodeLibraryDB:
    """SQLite-backed storage for Code Forge code units and search metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts_enabled = False
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS code_text (
                    content_hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    run_id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT
                );

                CREATE TABLE IF NOT EXISTS code_units (
                    id TEXT PRIMARY KEY,
                    language TEXT NOT NULL,
                    unit_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    qualified_name TEXT,
                    file_path TEXT NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    col_start INTEGER,
                    col_end INTEGER,
                    content_hash TEXT,
                    parent_id TEXT,
                    run_id TEXT,
                    complexity REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(content_hash) REFERENCES code_text(content_hash)
                );

                CREATE TABLE IF NOT EXISTS file_records (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    analysis_version INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id TEXT NOT NULL,
                    child_id TEXT NOT NULL,
                    rel_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(parent_id, child_id, rel_type)
                );

                CREATE TABLE IF NOT EXISTS code_fingerprints (
                    unit_id TEXT PRIMARY KEY,
                    normalized_hash TEXT,
                    simhash64 TEXT,
                    token_count INTEGER,
                    fingerprint_version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(unit_id) REFERENCES code_units(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS code_search (
                    unit_id TEXT PRIMARY KEY,
                    search_text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(unit_id) REFERENCES code_units(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_code_units_file_path
                    ON code_units(file_path);
                CREATE INDEX IF NOT EXISTS idx_code_units_qualified_name
                    ON code_units(qualified_name);
                CREATE INDEX IF NOT EXISTS idx_code_units_content_hash
                    ON code_units(content_hash);
                CREATE INDEX IF NOT EXISTS idx_code_units_language
                    ON code_units(language);
                CREATE INDEX IF NOT EXISTS idx_code_units_unit_type
                    ON code_units(unit_type);

                CREATE INDEX IF NOT EXISTS idx_code_fp_norm_hash
                    ON code_fingerprints(normalized_hash);
                CREATE INDEX IF NOT EXISTS idx_code_fp_simhash
                    ON code_fingerprints(simhash64);
                CREATE INDEX IF NOT EXISTS idx_code_fp_token_count
                    ON code_fingerprints(token_count);

                CREATE INDEX IF NOT EXISTS idx_relationships_parent_type
                    ON relationships(parent_id, rel_type);
                CREATE INDEX IF NOT EXISTS idx_relationships_child_type
                    ON relationships(child_id, rel_type);
                CREATE INDEX IF NOT EXISTS idx_relationships_type
                    ON relationships(rel_type);
                """
            )

            self._ensure_column(conn, "code_units", "complexity", "REAL")
            self._ensure_column(conn, "code_units", "language", "TEXT")

            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS code_units_fts
                    USING fts5(unit_id UNINDEXED, text)
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                # SQLite build may not include FTS5; code_search table remains usable.
                self._fts_enabled = False

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row[1] for row in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def create_run(
        self,
        root_path: str,
        mode: str,
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        run_id = run_id or hashlib.sha256(
            f"{root_path}|{mode}|{_utc_now()}".encode("utf-8")
        ).hexdigest()[:16]
        payload = json.dumps(config or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_runs (run_id, root_path, mode, created_at, config_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, root_path, mode, _utc_now(), payload),
            )
        return run_id

    def add_text(self, content: str) -> str:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO code_text (content_hash, content)
                VALUES (?, ?)
                """,
                (content_hash, content),
            )
        return content_hash

    def make_unit_id(self, unit: CodeUnit) -> str:
        parts = [
            unit.language,
            unit.unit_type,
            unit.file_path,
            unit.qualified_name or unit.name,
            str(unit.line_start or ""),
            str(unit.line_end or ""),
            str(unit.col_start or ""),
            str(unit.col_end or ""),
            unit.content_hash or "",
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _build_search_text(self, conn: sqlite3.Connection, unit: CodeUnit) -> str:
        parts = [
            unit.language,
            unit.unit_type,
            unit.name,
            unit.qualified_name or "",
            unit.file_path,
        ]
        snippet = unit.semantic_text
        if snippet is None and unit.content_hash:
            row = conn.execute(
                "SELECT content FROM code_text WHERE content_hash = ?",
                (unit.content_hash,),
            ).fetchone()
            if row:
                snippet = str(row[0])
        if snippet:
            parts.append(snippet[:3000])
        return "\n".join([p for p in parts if p]).strip()

    def _upsert_fingerprint(self, conn: sqlite3.Connection, unit_id: str, unit: CodeUnit) -> None:
        conn.execute(
            """
            INSERT INTO code_fingerprints (
                unit_id, normalized_hash, simhash64, token_count, fingerprint_version, created_at
            ) VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(unit_id) DO UPDATE SET
                normalized_hash=excluded.normalized_hash,
                simhash64=excluded.simhash64,
                token_count=excluded.token_count,
                fingerprint_version=excluded.fingerprint_version,
                created_at=excluded.created_at
            """,
            (
                unit_id,
                unit.normalized_hash,
                unit.simhash64,
                unit.token_count,
                _utc_now(),
            ),
        )

    def _upsert_search_index(self, conn: sqlite3.Connection, unit_id: str, search_text: str) -> None:
        conn.execute(
            """
            INSERT INTO code_search (unit_id, search_text, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(unit_id) DO UPDATE SET
                search_text=excluded.search_text,
                created_at=excluded.created_at
            """,
            (unit_id, search_text, _utc_now()),
        )

        if self._fts_enabled:
            conn.execute("DELETE FROM code_units_fts WHERE unit_id = ?", (unit_id,))
            conn.execute(
                "INSERT INTO code_units_fts (unit_id, text) VALUES (?, ?)",
                (unit_id, search_text),
            )

    def add_unit(self, unit: CodeUnit) -> str:
        unit_id = self.make_unit_id(unit)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO code_units (
                    id, language, unit_type, name, qualified_name, file_path,
                    line_start, line_end, col_start, col_end, content_hash,
                    parent_id, run_id, complexity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    line_start=COALESCE(code_units.line_start, excluded.line_start),
                    line_end=COALESCE(code_units.line_end, excluded.line_end),
                    col_start=COALESCE(code_units.col_start, excluded.col_start),
                    col_end=COALESCE(code_units.col_end, excluded.col_end),
                    content_hash=COALESCE(code_units.content_hash, excluded.content_hash),
                    parent_id=COALESCE(code_units.parent_id, excluded.parent_id),
                    run_id=COALESCE(code_units.run_id, excluded.run_id),
                    complexity=COALESCE(code_units.complexity, excluded.complexity)
                """,
                (
                    unit_id,
                    unit.language,
                    unit.unit_type,
                    unit.name,
                    unit.qualified_name,
                    unit.file_path,
                    unit.line_start,
                    unit.line_end,
                    unit.col_start,
                    unit.col_end,
                    unit.content_hash,
                    unit.parent_id,
                    unit.run_id,
                    unit.complexity,
                    _utc_now(),
                ),
            )

            self._upsert_fingerprint(conn, unit_id, unit)
            search_text = self._build_search_text(conn, unit)
            self._upsert_search_index(conn, unit_id, search_text)

        return unit_id

    def should_process_file(self, file_path: str, content_hash: str, analysis_version: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash, analysis_version FROM file_records WHERE file_path = ?",
                (file_path,),
            ).fetchone()
        if not row:
            return True
        if row["content_hash"] != content_hash:
            return True
        return row["analysis_version"] < analysis_version

    def update_file_record(self, file_path: str, content_hash: str, analysis_version: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO file_records (file_path, content_hash, analysis_version, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    analysis_version=excluded.analysis_version,
                    updated_at=excluded.updated_at
                """,
                (file_path, content_hash, analysis_version, _utc_now()),
            )

    def get_unit(self, unit_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cu.*, fp.normalized_hash, fp.simhash64, fp.token_count
                FROM code_units cu
                LEFT JOIN code_fingerprints fp ON fp.unit_id = cu.id
                WHERE cu.id = ?
                """,
                (unit_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_text(self, content_hash: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM code_text WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        return row[0] if row else None

    def add_relationship(self, parent_id: str, child_id: str, rel_type: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO relationships (parent_id, child_id, rel_type, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (parent_id, child_id, rel_type, _utc_now()),
            )

    def relationship_counts(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rel_type, COUNT(*) AS c
                FROM relationships
                GROUP BY rel_type
                ORDER BY c DESC
                """
            ).fetchall()
        return {str(r["rel_type"]): int(r["c"]) for r in rows}

    def list_relationships(
        self,
        rel_type: Optional[str] = None,
        limit: int = 5000,
    ) -> list[Dict[str, Any]]:
        where = ""
        params: list[Any] = []
        if rel_type:
            where = "WHERE r.rel_type = ?"
            params.append(rel_type)
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    r.rel_type,
                    r.parent_id,
                    r.child_id,
                    p.qualified_name AS parent_qualified_name,
                    p.file_path AS parent_file_path,
                    p.unit_type AS parent_unit_type,
                    c.qualified_name AS child_qualified_name,
                    c.file_path AS child_file_path,
                    c.unit_type AS child_unit_type
                FROM relationships r
                LEFT JOIN code_units p ON p.id = r.parent_id
                LEFT JOIN code_units c ON c.id = r.child_id
                {where}
                ORDER BY r.id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(r) for r in rows]

    def module_dependency_graph(
        self,
        rel_types: Optional[list[str]] = None,
        limit_edges: int = 20000,
    ) -> Dict[str, Any]:
        rel_types = rel_types or ["imports", "calls", "uses"]
        rel_types = [r for r in rel_types if r]
        if not rel_types:
            return {"nodes": [], "edges": [], "summary": {"edge_count": 0, "node_count": 0}}

        placeholders = ",".join(["?"] * len(rel_types))
        params: list[Any] = list(rel_types)
        params.append(max(1, int(limit_edges)))

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    r.rel_type,
                    p.file_path AS parent_file,
                    c.file_path AS child_file,
                    p.qualified_name AS parent_qn,
                    c.qualified_name AS child_qn
                FROM relationships r
                JOIN code_units p ON p.id = r.parent_id
                JOIN code_units c ON c.id = r.child_id
                WHERE r.rel_type IN ({placeholders})
                ORDER BY r.id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

        node_map: Dict[str, Dict[str, Any]] = {}
        edge_counts: Dict[tuple[str, str, str], int] = {}
        for row in rows:
            src = str(row["parent_file"] or "__unknown__")
            dst = str(row["child_file"] or "__unknown__")
            rel = str(row["rel_type"] or "unknown")
            if src not in node_map:
                node_map[src] = {"id": src, "path": src}
            if dst not in node_map:
                node_map[dst] = {"id": dst, "path": dst}
            key = (src, dst, rel)
            edge_counts[key] = edge_counts.get(key, 0) + 1

        edges = [
            {"source": src, "target": dst, "rel_type": rel, "weight": weight}
            for (src, dst, rel), weight in edge_counts.items()
        ]
        edges.sort(key=lambda e: (e["rel_type"], -int(e["weight"]), e["source"], e["target"]))

        return {
            "nodes": list(node_map.values()),
            "edges": edges,
            "summary": {
                "node_count": len(node_map),
                "edge_count": len(edges),
                "relationship_rows": len(rows),
                "rel_types": rel_types,
            },
        }

    def iter_units(self, limit: int = 1000) -> Iterable[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT cu.*, fp.normalized_hash, fp.simhash64, fp.token_count
                FROM code_units cu
                LEFT JOIN code_fingerprints fp ON fp.unit_id = cu.id
                ORDER BY cu.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        for row in rows:
            yield dict(row)

    def find_unit_by_qualified_name(self, qualified_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT cu.*, fp.normalized_hash, fp.simhash64, fp.token_count
                FROM code_units cu
                LEFT JOIN code_fingerprints fp ON fp.unit_id = cu.id
                WHERE cu.qualified_name = ?
                ORDER BY cu.created_at DESC
                LIMIT 1
                """,
                (qualified_name,),
            ).fetchone()
        return dict(row) if row else None

    def get_children(
        self,
        parent_id: str,
        rel_type: Optional[str] = "contains",
        limit: int = 500,
    ) -> list[Dict[str, Any]]:
        where = "r.parent_id = ?"
        params: list[Any] = [parent_id]
        if rel_type is not None:
            where += " AND r.rel_type = ?"
            params.append(rel_type)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT cu.*, fp.normalized_hash, fp.simhash64, fp.token_count, r.rel_type
                FROM relationships r
                JOIN code_units cu ON cu.id = r.child_id
                LEFT JOIN code_fingerprints fp ON fp.unit_id = cu.id
                WHERE {where}
                ORDER BY cu.created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_parents(
        self,
        child_id: str,
        rel_type: Optional[str] = "contains",
        limit: int = 500,
    ) -> list[Dict[str, Any]]:
        where = "r.child_id = ?"
        params: list[Any] = [child_id]
        if rel_type is not None:
            where += " AND r.rel_type = ?"
            params.append(rel_type)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT cu.*, fp.normalized_hash, fp.simhash64, fp.token_count, r.rel_type
                FROM relationships r
                JOIN code_units cu ON cu.id = r.parent_id
                LEFT JOIN code_fingerprints fp ON fp.unit_id = cu.id
                WHERE {where}
                ORDER BY cu.created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_duplicate_units(
        self,
        min_occurrences: int = 2,
        limit_groups: int = 200,
    ) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            groups = conn.execute(
                """
                SELECT content_hash, COUNT(*) AS occurrences
                FROM code_units
                WHERE content_hash IS NOT NULL
                GROUP BY content_hash
                HAVING COUNT(*) >= ?
                ORDER BY occurrences DESC
                LIMIT ?
                """,
                (max(2, min_occurrences), max(1, limit_groups)),
            ).fetchall()
            out: list[Dict[str, Any]] = []
            for grp in groups:
                content_hash = grp["content_hash"]
                rows = conn.execute(
                    """
                    SELECT id, unit_type, language, qualified_name, file_path, line_start, line_end
                    FROM code_units
                    WHERE content_hash = ?
                    ORDER BY file_path, line_start
                    """,
                    (content_hash,),
                ).fetchall()
                out.append(
                    {
                        "content_hash": content_hash,
                        "occurrences": int(grp["occurrences"]),
                        "units": [dict(r) for r in rows],
                    }
                )
        return out

    def list_normalized_duplicates(
        self,
        min_occurrences: int = 2,
        limit_groups: int = 200,
    ) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            groups = conn.execute(
                """
                SELECT fp.normalized_hash, COUNT(*) AS occurrences
                FROM code_fingerprints fp
                WHERE fp.normalized_hash IS NOT NULL
                GROUP BY fp.normalized_hash
                HAVING COUNT(*) >= ?
                ORDER BY occurrences DESC
                LIMIT ?
                """,
                (max(2, min_occurrences), max(1, limit_groups)),
            ).fetchall()

            out: list[Dict[str, Any]] = []
            for grp in groups:
                normalized_hash = grp["normalized_hash"]
                rows = conn.execute(
                    """
                    SELECT cu.id, cu.unit_type, cu.language, cu.qualified_name, cu.file_path,
                           cu.line_start, cu.line_end, fp.simhash64, fp.token_count
                    FROM code_units cu
                    JOIN code_fingerprints fp ON fp.unit_id = cu.id
                    WHERE fp.normalized_hash = ?
                    ORDER BY cu.file_path, cu.line_start
                    """,
                    (normalized_hash,),
                ).fetchall()
                out.append(
                    {
                        "normalized_hash": normalized_hash,
                        "occurrences": int(grp["occurrences"]),
                        "units": [dict(r) for r in rows],
                    }
                )
        return out

    def list_near_duplicates(
        self,
        max_hamming: int = 6,
        min_token_count: int = 20,
        limit_pairs: int = 200,
        max_units: int = 3000,
        language: Optional[str] = None,
        unit_type: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        where = ["fp.simhash64 IS NOT NULL", "COALESCE(fp.token_count, 0) >= ?"]
        params: list[Any] = [max(1, int(min_token_count))]

        if language:
            where.append("cu.language = ?")
            params.append(language)
        if unit_type:
            where.append("cu.unit_type = ?")
            params.append(unit_type)

        params.append(max(10, int(max_units)))

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT cu.id, cu.language, cu.unit_type, cu.qualified_name, cu.file_path,
                       cu.line_start, cu.line_end, fp.normalized_hash, fp.simhash64, fp.token_count
                FROM code_units cu
                JOIN code_fingerprints fp ON fp.unit_id = cu.id
                WHERE {' AND '.join(where)}
                ORDER BY COALESCE(fp.token_count, 0) DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

        records = [dict(r) for r in rows]
        out: list[Dict[str, Any]] = []
        threshold = max(0, int(max_hamming))

        for i in range(len(records)):
            left = records[i]
            left_hash = left.get("simhash64")
            if not left_hash:
                continue
            for j in range(i + 1, len(records)):
                right = records[j]
                right_hash = right.get("simhash64")
                if not right_hash:
                    continue
                if left["id"] == right["id"]:
                    continue
                if left.get("normalized_hash") and left.get("normalized_hash") == right.get("normalized_hash"):
                    continue

                dist = hamming_distance64(int(left_hash, 16), int(right_hash, 16))
                if dist > threshold:
                    continue

                out.append(
                    {
                        "distance": dist,
                        "similarity": round(1.0 - (dist / 64.0), 4),
                        "left": left,
                        "right": right,
                    }
                )
                if len(out) >= limit_pairs:
                    break
            if len(out) >= limit_pairs:
                break

        out.sort(key=lambda x: (x["distance"], -x["similarity"]))
        return out

    def _fallback_search_rows(
        self,
        conn: sqlite3.Connection,
        query_tokens: list[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        if not query_tokens:
            rows = conn.execute(
                """
                SELECT unit_id, search_text
                FROM code_search
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [(str(r["unit_id"]), 0.0) for r in rows]

        like_terms = [f"%{tok}%" for tok in query_tokens]
        where = " OR ".join(["search_text LIKE ?" for _ in like_terms])
        rows = conn.execute(
            f"""
            SELECT unit_id, search_text
            FROM code_search
            WHERE {where}
            LIMIT ?
            """,
            tuple(like_terms + [limit]),
        ).fetchall()
        return [(str(r["unit_id"]), 0.0) for r in rows]

    def semantic_search(
        self,
        query: str,
        limit: int = 20,
        language: Optional[str] = None,
        unit_type: Optional[str] = None,
        min_score: float = 0.05,
    ) -> list[Dict[str, Any]]:
        query_tokens = tokenize_code_text(query)
        with self._connect() as conn:
            candidates: list[tuple[str, float]] = []
            if self._fts_enabled and query_tokens:
                fts_query = " OR ".join([f"{tok}*" for tok in query_tokens])
                try:
                    rows = conn.execute(
                        """
                        SELECT unit_id, bm25(code_units_fts) AS rank
                        FROM code_units_fts
                        WHERE code_units_fts MATCH ?
                        LIMIT ?
                        """,
                        (fts_query, max(limit * 5, 50)),
                    ).fetchall()
                    for row in rows:
                        rank = float(row["rank"])
                        score = 1.0 / (1.0 + max(rank, 0.0))
                        candidates.append((str(row["unit_id"]), score))
                except sqlite3.OperationalError:
                    # Invalid query string and/or tokenizer mismatch.
                    candidates = []

            if not candidates:
                candidates = self._fallback_search_rows(conn, query_tokens, max(limit * 5, 50))

            seen: set[str] = set()
            out: list[Dict[str, Any]] = []
            for unit_id, fts_score in candidates:
                if unit_id in seen:
                    continue
                seen.add(unit_id)

                row = conn.execute(
                    """
                    SELECT cu.*, fp.normalized_hash, fp.simhash64, fp.token_count, cs.search_text
                    FROM code_units cu
                    LEFT JOIN code_fingerprints fp ON fp.unit_id = cu.id
                    LEFT JOIN code_search cs ON cs.unit_id = cu.id
                    WHERE cu.id = ?
                    """,
                    (unit_id,),
                ).fetchone()
                if row is None:
                    continue

                rec = dict(row)
                if language and rec.get("language") != language:
                    continue
                if unit_type and rec.get("unit_type") != unit_type:
                    continue

                search_text = str(rec.get("search_text") or "")
                unit_tokens = tokenize_code_text(search_text)
                lexical = token_jaccard(query_tokens, unit_tokens)
                substring_boost = 0.0
                if query and query.lower() in search_text.lower():
                    substring_boost = 0.1

                score = (0.65 * fts_score) + (0.35 * lexical) + substring_boost
                if score < min_score:
                    continue

                rec["semantic_score"] = round(score, 4)
                rec["fts_score"] = round(fts_score, 4)
                rec["lexical_score"] = round(lexical, 4)
                rec["search_preview"] = search_text[:280]
                out.append(rec)

                if len(out) >= limit:
                    break

        out.sort(key=lambda x: x.get("semantic_score", 0.0), reverse=True)
        return out

    def trace_contains(
        self,
        root_unit_id: str,
        max_depth: int = 3,
        max_nodes: int = 300,
    ) -> Dict[str, Any]:
        root = self.get_unit(root_unit_id)
        if not root:
            return {"root": None, "nodes": [], "edges": []}

        nodes: Dict[str, Dict[str, Any]] = {root_unit_id: root}
        edges: list[Dict[str, Any]] = []
        frontier: list[tuple[str, int]] = [(root_unit_id, 0)]
        seen: set[str] = {root_unit_id}

        while frontier and len(nodes) < max_nodes:
            node_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            children = self.get_children(node_id, rel_type="contains", limit=max_nodes)
            for child in children:
                cid = str(child.get("id"))
                edges.append({"from": node_id, "to": cid, "rel_type": "contains"})
                if cid in seen:
                    continue
                seen.add(cid)
                nodes[cid] = child
                frontier.append((cid, depth + 1))
                if len(nodes) >= max_nodes:
                    break

        return {
            "root": root_unit_id,
            "nodes": list(nodes.values()),
            "edges": edges,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        }

    def count_units(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM code_units").fetchone()
        return int(row["c"]) if row else 0

    def file_metrics(self, limit: int = 100000) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    cu.file_path AS file_path,
                    GROUP_CONCAT(DISTINCT cu.language) AS languages,
                    COUNT(*) AS unit_count,
                    SUM(CASE WHEN cu.unit_type = 'module' THEN 1 ELSE 0 END) AS module_units,
                    SUM(CASE WHEN cu.unit_type = 'class' THEN 1 ELSE 0 END) AS class_units,
                    SUM(CASE WHEN cu.unit_type IN ('function', 'method') THEN 1 ELSE 0 END) AS callable_units,
                    AVG(COALESCE(cu.complexity, 0.0)) AS avg_complexity,
                    MAX(COALESCE(cu.complexity, 0.0)) AS max_complexity,
                    SUM(COALESCE(fp.token_count, 0)) AS token_count_sum,
                    COUNT(DISTINCT COALESCE(fp.normalized_hash, cu.id)) AS unique_fingerprint_count
                FROM code_units cu
                LEFT JOIN code_fingerprints fp ON fp.unit_id = cu.id
                GROUP BY cu.file_path
                ORDER BY unit_count DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()

        out: list[Dict[str, Any]] = []
        for row in rows:
            rec = dict(row)
            languages = [part for part in str(rec.get("languages") or "").split(",") if part]
            rec["languages"] = sorted(set(languages))
            rec["unit_count"] = _safe_int(rec.get("unit_count"), 0)
            rec["module_units"] = _safe_int(rec.get("module_units"), 0)
            rec["class_units"] = _safe_int(rec.get("class_units"), 0)
            rec["callable_units"] = _safe_int(rec.get("callable_units"), 0)
            rec["token_count_sum"] = _safe_int(rec.get("token_count_sum"), 0)
            rec["unique_fingerprint_count"] = _safe_int(rec.get("unique_fingerprint_count"), 0)
            rec["avg_complexity"] = float(rec.get("avg_complexity") or 0.0)
            rec["max_complexity"] = float(rec.get("max_complexity") or 0.0)
            out.append(rec)
        return out

    def count_units_by_type(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT unit_type, COUNT(*) AS c
                FROM code_units
                GROUP BY unit_type
                ORDER BY c DESC
                """
            ).fetchall()
        return {str(r["unit_type"]): int(r["c"]) for r in rows}

    def count_units_by_language(self) -> Dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT language, COUNT(*) AS c
                FROM code_units
                GROUP BY language
                ORDER BY c DESC
                """
            ).fetchall()
        return {str(r["language"]): int(r["c"]) for r in rows}

    def latest_runs(self, limit: int = 10) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, root_path, mode, created_at, config_json
                FROM ingestion_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["config"] = json.loads(item.pop("config_json") or "{}")
            except Exception:
                item["config"] = {}
                item.pop("config_json", None)
            out.append(item)
        return out
