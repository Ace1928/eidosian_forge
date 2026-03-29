from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from word_forge.database.database_manager import DBManager
from word_forge.multilingual.fasttext_ingestor import FastTextIngestor
from word_forge.multilingual.runtime import append_history, db_counts, read_json, source_signature, write_json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_fasttext_report(
    *,
    source_path: Path,
    lang: str,
    db_path: Path,
    vector_db_path: Path,
    limit: int | None = None,
    bootstrap_lang: str | None = None,
    top_k: int = 1,
    min_score: float = 0.55,
    apply: bool = False,
) -> dict[str, Any]:
    db_manager = DBManager(db_path=db_path)
    ingestor = FastTextIngestor(db_manager=db_manager, storage_path=vector_db_path)
    before_words = db_counts(db_path)
    before_vectors = ingestor.vector_counts()

    ingest_result = ingestor.ingest_vectors(str(source_path), lang=lang, limit=limit or 10000)
    if ingest_result.is_failure:
        raise RuntimeError(str(ingest_result.error))

    bootstrap_result: dict[str, Any] | None = None
    if bootstrap_lang:
        result = ingestor.bootstrap_translations(
            lang,
            bootstrap_lang,
            top_k=max(1, int(top_k)),
            min_score=float(min_score),
            apply=apply,
        )
        if result.is_failure:
            raise RuntimeError(str(result.error))
        bootstrap_result = result.value

    after_words = db_counts(db_path)
    after_vectors = ingestor.vector_counts()
    return {
        "contract": "eidos.word_forge.fasttext_ingest.v1",
        "generated_at": now_iso(),
        "source_path": str(source_path),
        "source_signature": source_signature(source_path),
        "lang": lang,
        "db_path": str(db_path),
        "vector_db_path": str(vector_db_path),
        "limit": limit,
        "bootstrap_lang": bootstrap_lang,
        "top_k": top_k,
        "min_score": min_score,
        "apply": apply,
        "before": {"word": before_words, "fasttext": before_vectors},
        "after": {"word": after_words, "fasttext": after_vectors},
        "deltas": {
            "vector_delta": after_vectors["vector_count"] - before_vectors["vector_count"],
            "candidate_delta": after_vectors["candidate_count"] - before_vectors["candidate_count"],
            "applied_delta": after_vectors["applied_count"] - before_vectors["applied_count"],
            "translation_delta": after_words["translation_count"] - before_words["translation_count"],
            "base_aligned_delta": after_words["base_aligned_count"] - before_words["base_aligned_count"],
        },
        "bootstrap": bootstrap_result or {},
    }


def render_fasttext_markdown(report: dict[str, Any]) -> str:
    after_word = (report.get("after") or {}).get("word") or {}
    after_fast = (report.get("after") or {}).get("fasttext") or {}
    deltas = report.get("deltas") or {}
    bootstrap = report.get("bootstrap") or {}
    lines = [
        "# Word Forge FastText Ingest",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Source path: `{report.get('source_path')}`",
        f"- Language: `{report.get('lang')}`",
        f"- Bootstrap lang: `{report.get('bootstrap_lang')}`",
        f"- Applied: `{report.get('apply')}`",
        f"- Vector delta: `{deltas.get('vector_delta', 0)}`",
        f"- Candidate delta: `{deltas.get('candidate_delta', 0)}`",
        f"- Applied delta: `{deltas.get('applied_delta', 0)}`",
        f"- Translation delta: `{deltas.get('translation_delta', 0)}`",
        f"- Base aligned delta: `{deltas.get('base_aligned_delta', 0)}`",
        "",
        "## FastText Store",
        "",
        f"- Vectors: `{after_fast.get('vector_count', 0)}`",
        f"- Candidate rows: `{after_fast.get('candidate_count', 0)}`",
        f"- Applied rows: `{after_fast.get('applied_count', 0)}`",
        f"- Languages: `{after_fast.get('language_count', 0)}`",
        "",
        "## Word Forge Totals",
        "",
        f"- Lexemes: `{after_word.get('lexeme_count', 0)}`",
        f"- Translations: `{after_word.get('translation_count', 0)}`",
        f"- Base aligned: `{after_word.get('base_aligned_count', 0)}`",
        "",
        "## Top Candidates",
        "",
        "| Source | Target | Score | Rank |",
        "| --- | --- | ---: | ---: |",
    ]
    rows = bootstrap.get("top_candidates") or []
    if rows:
        for row in rows[:20]:
            lines.append(
                f"| {row.get('src_term')} | {row.get('dst_term')} | {row.get('score')} | {row.get('rank')} |"
            )
    else:
        lines.append("| none | none | 0 | 0 |")
    return "\n".join(lines) + "\n"


def write_fasttext_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_fasttext_ingest"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_fasttext_ingest_{stamp}.json"
    md_path = report_dir / f"word_forge_fasttext_ingest_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    write_json(json_path, report)
    md_path.write_text(render_fasttext_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_fasttext_ingest(
    *,
    repo_root: Path,
    source_path: Path,
    lang: str,
    db_path: Path,
    vector_db_path: Path,
    limit: int | None = None,
    bootstrap_lang: str | None = None,
    top_k: int = 1,
    min_score: float = 0.55,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_fasttext_ingest_status.json"
    history_path = runtime_dir / "word_forge_fasttext_ingest_history.jsonl"
    signature = source_signature(source_path)
    previous = read_json(status_path)
    status_payload = {
        "contract": "eidos.word_forge.fasttext_ingest.status.v1",
        "status": "running",
        "phase": "ingesting",
        "source_path": str(source_path),
        "source_signature": signature,
        "lang": lang,
        "db_path": str(db_path),
        "vector_db_path": str(vector_db_path),
        "bootstrap_lang": bootstrap_lang,
        "top_k": top_k,
        "min_score": min_score,
        "apply": apply,
        "limit": limit,
        "force": force,
        "started_at": now_iso(),
    }
    write_json(status_path, status_payload)
    try:
        if not force and previous.get("source_signature") == signature and previous.get("status") == "completed":
            status_payload.update({"status": "skipped", "phase": "unchanged", "finished_at": now_iso()})
            write_json(status_path, status_payload)
            append_history(history_path, status_payload)
            return {"status": status_payload, "report": None, "artifacts": {}}

        report = build_fasttext_report(
            source_path=source_path,
            lang=lang,
            db_path=db_path,
            vector_db_path=vector_db_path,
            limit=limit,
            bootstrap_lang=bootstrap_lang,
            top_k=top_k,
            min_score=min_score,
            apply=apply,
        )
        artifacts = write_fasttext_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "vector_delta": report["deltas"]["vector_delta"],
                "candidate_delta": report["deltas"]["candidate_delta"],
                "applied_delta": report["deltas"]["applied_delta"],
                "translation_delta": report["deltas"]["translation_delta"],
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
