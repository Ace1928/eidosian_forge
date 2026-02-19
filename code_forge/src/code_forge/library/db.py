from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


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


class CodeLibraryDB:
    """SQLite-backed storage for Code Forge code units and text blobs."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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

                CREATE INDEX IF NOT EXISTS idx_code_units_file_path
                    ON code_units(file_path);
                CREATE INDEX IF NOT EXISTS idx_code_units_qualified_name
                    ON code_units(qualified_name);
                CREATE INDEX IF NOT EXISTS idx_code_units_content_hash
                    ON code_units(content_hash);
                """
            )
            self._ensure_column(conn, "code_units", "complexity", "REAL")

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
                "SELECT * FROM code_units WHERE id = ?", (unit_id,)
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

    def iter_units(self, limit: int = 1000) -> Iterable[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM code_units ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        for row in rows:
            yield dict(row)

    def find_unit_by_qualified_name(self, qualified_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM code_units
                WHERE qualified_name = ?
                ORDER BY created_at DESC
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
                SELECT cu.*, r.rel_type
                FROM relationships r
                JOIN code_units cu ON cu.id = r.child_id
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
                SELECT cu.*, r.rel_type
                FROM relationships r
                JOIN code_units cu ON cu.id = r.parent_id
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
                    SELECT id, unit_type, qualified_name, file_path, line_start, line_end
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
