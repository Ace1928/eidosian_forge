from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from word_forge.database.database_manager import DBManager
from word_forge.linguistics.morphology import MorphologyManager
from word_forge.multilingual.runtime import append_history, read_json, write_json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _counts(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "lexeme_count": 0,
            "lexeme_morpheme_count": 0,
            "decomposed_lexeme_count": 0,
            "morpheme_count": 0,
            "top_languages": [],
        }
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        lexeme_count = int(conn.execute("SELECT COUNT(*) FROM lexemes").fetchone()[0])
        morpheme_count = int(conn.execute("SELECT COUNT(*) FROM morphemes").fetchone()[0])
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lexeme_morphemes (
                lexeme_id INTEGER NOT NULL,
                morpheme_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY(lexeme_id, position)
            )
            """
        )
        lexeme_morpheme_count = int(conn.execute("SELECT COUNT(*) FROM lexeme_morphemes").fetchone()[0])
        decomposed_lexeme_count = int(conn.execute("SELECT COUNT(DISTINCT lexeme_id) FROM lexeme_morphemes").fetchone()[0])
        top_languages = conn.execute(
            """
            SELECT l.lang AS lang, COUNT(DISTINCT lm.lexeme_id) AS count
            FROM lexeme_morphemes lm
            JOIN lexemes l ON l.id = lm.lexeme_id
            GROUP BY l.lang
            ORDER BY count DESC, l.lang ASC
            LIMIT 12
            """
        ).fetchall()
    return {
        "lexeme_count": lexeme_count,
        "lexeme_morpheme_count": lexeme_morpheme_count,
        "decomposed_lexeme_count": decomposed_lexeme_count,
        "morpheme_count": morpheme_count,
        "top_languages": [{"lang": row["lang"], "count": int(row["count"])} for row in top_languages],
    }


def build_polyglot_report(*, db_path: Path, lang: str | None = None, limit: int | None = None) -> dict[str, Any]:
    db = DBManager(db_path=db_path)
    manager = MorphologyManager(db_manager=db)
    before = _counts(db_path)
    processed = 0
    decomposed = 0
    samples: list[dict[str, Any]] = []
    query = "SELECT id, lemma, lang, base_term FROM lexemes"
    params: list[Any] = []
    if lang:
        query += " WHERE lang = ?"
        params.append(lang)
    query += " ORDER BY last_refreshed DESC"
    if limit:
        query += " LIMIT ?"
        params.append(max(1, int(limit)))
    rows = db.execute_query(query, tuple(params))
    for row in rows:
        lexeme_id = int(row["id"])
        lemma = str(row["lemma"] or "")
        parts = [part for part in manager.decompose(lemma) if str(part).strip()]
        if not parts:
            continue
        morpheme_ids = [manager.upsert_morpheme(part) for part in parts]
        manager.set_lexeme_morphemes(lexeme_id, morpheme_ids)
        processed += 1
        if len(parts) > 1:
            decomposed += 1
        if len(samples) < 20:
            samples.append({"lemma": lemma, "lang": row["lang"], "base_term": row["base_term"], "morphemes": parts})
    after = _counts(db_path)
    return {
        "contract": "eidos.word_forge.polyglot_decomposition.v1",
        "generated_at": now_iso(),
        "db_path": str(db_path),
        "lang": lang,
        "limit": limit,
        "before": before,
        "after": after,
        "deltas": {
            "lexeme_morpheme_delta": after["lexeme_morpheme_count"] - before["lexeme_morpheme_count"],
            "decomposed_lexeme_delta": after["decomposed_lexeme_count"] - before["decomposed_lexeme_count"],
            "morpheme_delta": after["morpheme_count"] - before["morpheme_count"],
        },
        "processed_lexemes": processed,
        "multi_part_lexemes": decomposed,
        "samples": samples,
    }


def render_polyglot_markdown(report: dict[str, Any]) -> str:
    after = report.get("after") or {}
    deltas = report.get("deltas") or {}
    lines = [
        "# Word Forge Polyglot Decomposition",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Language filter: `{report.get('lang')}`",
        f"- Processed lexemes: `{report.get('processed_lexemes', 0)}`",
        f"- Multi-part lexemes: `{report.get('multi_part_lexemes', 0)}`",
        f"- Lexeme morpheme delta: `{deltas.get('lexeme_morpheme_delta', 0)}`",
        f"- Decomposed lexeme delta: `{deltas.get('decomposed_lexeme_delta', 0)}`",
        f"- Morpheme delta: `{deltas.get('morpheme_delta', 0)}`",
        "",
        "## Totals",
        "",
        f"- Lexemes: `{after.get('lexeme_count', 0)}`",
        f"- Decomposed lexemes: `{after.get('decomposed_lexeme_count', 0)}`",
        f"- Lexeme morpheme rows: `{after.get('lexeme_morpheme_count', 0)}`",
        f"- Morphemes: `{after.get('morpheme_count', 0)}`",
        "",
        "## Samples",
        "",
        "| Lemma | Lang | Base | Morphemes |",
        "| --- | --- | --- | --- |",
    ]
    samples = report.get("samples") or []
    if samples:
        for row in samples[:20]:
            lines.append(
                f"| {row.get('lemma')} | {row.get('lang')} | {row.get('base_term') or ''} | {', '.join(row.get('morphemes') or [])} |"
            )
    else:
        lines.append("| none | none | none | none |")
    return "\n".join(lines) + "\n"


def write_polyglot_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_polyglot"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_polyglot_{stamp}.json"
    md_path = report_dir / f"word_forge_polyglot_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    write_json(json_path, report)
    md_path.write_text(render_polyglot_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_polyglot_decomposition(*, repo_root: Path, db_path: Path, lang: str | None = None, limit: int | None = None, force: bool = False) -> dict[str, Any]:
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_polyglot_status.json"
    history_path = runtime_dir / "word_forge_polyglot_history.jsonl"
    previous = read_json(status_path)
    status_payload = {
        "contract": "eidos.word_forge.polyglot.status.v1",
        "status": "running",
        "phase": "decomposing",
        "db_path": str(db_path),
        "lang": lang,
        "limit": limit,
        "force": force,
        "started_at": now_iso(),
    }
    write_json(status_path, status_payload)
    try:
        if not force and previous.get("status") == "completed" and previous.get("lang") == lang and previous.get("limit") == limit:
            status_payload.update({"status": "skipped", "phase": "unchanged", "finished_at": now_iso()})
            write_json(status_path, status_payload)
            append_history(history_path, status_payload)
            return {"status": status_payload, "report": None, "artifacts": {}}
        report = build_polyglot_report(db_path=db_path, lang=lang, limit=limit)
        artifacts = write_polyglot_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "processed_lexemes": report["processed_lexemes"],
                "multi_part_lexemes": report["multi_part_lexemes"],
                "decomposed_lexeme_delta": report["deltas"]["decomposed_lexeme_delta"],
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update({"status": "failed", "phase": "failed", "finished_at": now_iso(), "error": str(exc)})
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise
