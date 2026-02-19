#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[1]))).resolve()
for extra in (FORGE_ROOT / "lib", FORGE_ROOT / "code_forge" / "src", FORGE_ROOT):
    extra_str = str(extra)
    if extra.exists() and extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB
from eidosian_core import eidosian
from eidosian_core.ports import get_service_port

SUPPORTED_TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".rst",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".sql",
}

EXCLUDE_SEGMENTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
}

TEXT_PROMPT = """You are generating strict JSON community reports from source text.
Use only facts present in the text. Never use placeholders or template content.
Return exactly one JSON object with keys: title, summary, rating, rating_explanation, findings.
"""

GRAPH_PROMPT = """You are generating strict JSON community reports from graph context.
Use only grounded facts and return one strict JSON object with keys:
title, summary, rating, rating_explanation, findings.
"""


@dataclass
class StagedRecord:
    doc_id: str
    source_path: str
    kind: str
    sha256: str
    bytes: int
    chars: int
    staged_path: str
    simhash: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_binary_blob(data: bytes) -> bool:
    return b"\x00" in data


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _kind_for_path(path: Path) -> str:
    low = path.as_posix().lower()
    if low.endswith("memory_data.json") or "tiered_memory" in low:
        return "memory"
    if low.endswith("data/kb.json"):
        return "knowledge"
    if path.suffix.lower() in {".py", ".ts", ".tsx", ".js", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp"}:
        return "code"
    if path.suffix.lower() in {".md", ".rst", ".txt"}:
        return "docs"
    return "config"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())


def _simhash64(text: str) -> int:
    tokens = _tokenize(text)
    if not tokens:
        return 0
    weights = Counter(tokens)
    acc = [0] * 64
    for token, w in weights.items():
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            acc[bit] += w if (h >> bit) & 1 else -w
    out = 0
    for bit, val in enumerate(acc):
        if val >= 0:
            out |= 1 << bit
    return out


def _hamming_distance64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


@eidosian()
def extract_memory_records(memory_path: Path) -> list[dict[str, str]]:
    if not memory_path.exists():
        return []
    try:
        payload = json.loads(memory_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    out: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return out

    def _emit(namespace: str, item: Any, index: int) -> None:
        if not isinstance(item, dict):
            return
        content = str(item.get("content", "")).strip()
        if not content:
            return
        key = item.get("id") or item.get("key") or f"{namespace}_{index}"
        out.append({"id": f"{namespace}:{key}", "content": content})

    for namespace, value in payload.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                _emit(str(namespace), item, i)
        elif isinstance(value, dict):
            _emit(str(namespace), value, 0)
    return out


@eidosian()
def extract_kb_records(kb_path: Path) -> list[dict[str, str]]:
    if not kb_path.exists():
        return []
    try:
        payload = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, dict):
        return []
    out: list[dict[str, str]] = []
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        content = str(node.get("content", "")).strip()
        if not content:
            continue
        out.append({"id": str(node_id), "content": content})
    return out


def _iter_git_tracked_files(repo_root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    files: list[Path] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        files.append(repo_root / line)
    return files


@eidosian()
def group_exact_duplicates(records: list[StagedRecord]) -> list[dict[str, Any]]:
    groups: dict[str, list[StagedRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.sha256].append(rec)
    out = []
    for digest, bucket in groups.items():
        if len(bucket) < 2:
            continue
        out.append(
            {
                "sha256": digest,
                "occurrences": len(bucket),
                "documents": [r.doc_id for r in bucket],
                "paths": [r.source_path for r in bucket],
            }
        )
    out.sort(key=lambda row: row["occurrences"], reverse=True)
    return out


@eidosian()
def detect_near_duplicates(records: list[StagedRecord], max_hamming: int = 3) -> list[dict[str, Any]]:
    near: list[dict[str, Any]] = []
    buckets: dict[tuple[int, int], list[StagedRecord]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for rec in records:
        sim = int(rec.simhash, 16)
        for band in range(4):
            bucket_key = (band, (sim >> (band * 16)) & 0xFFFF)
            for prior in buckets[bucket_key]:
                pair_key = tuple(sorted((prior.doc_id, rec.doc_id)))
                if pair_key in seen_pairs:
                    continue
                prior_sim = int(prior.simhash, 16)
                dist = _hamming_distance64(sim, prior_sim)
                if dist > max_hamming:
                    continue
                min_chars = min(rec.chars, prior.chars)
                max_chars = max(rec.chars, prior.chars)
                if min_chars <= 0:
                    continue
                if (max_chars / min_chars) > 1.5:
                    continue
                seen_pairs.add(pair_key)
                near.append(
                    {
                        "doc_a": prior.doc_id,
                        "doc_b": rec.doc_id,
                        "path_a": prior.source_path,
                        "path_b": rec.source_path,
                        "hamming_distance": dist,
                    }
                )
            buckets[bucket_key].append(rec)
    near.sort(key=lambda row: row["hamming_distance"])
    return near


def _write_text_document(target: Path, source_path: str, kind: str, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"SOURCE_PATH: {source_path}\nKIND: {kind}\n\n{content.strip()}\n",
        encoding="utf-8",
    )


@eidosian()
def stage_repo_text_documents(
    repo_root: Path,
    stage_dir: Path,
    max_file_bytes: int,
    max_chars_per_doc: int,
) -> list[StagedRecord]:
    out: list[StagedRecord] = []
    for path in _iter_git_tracked_files(repo_root):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix and suffix not in SUPPORTED_TEXT_SUFFIXES:
            continue
        rel = path.resolve().relative_to(repo_root.resolve())
        if any(seg in EXCLUDE_SEGMENTS for seg in rel.parts):
            continue
        raw = path.read_bytes()
        if len(raw) > max_file_bytes:
            continue
        if _is_binary_blob(raw):
            continue
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        text = text[:max_chars_per_doc]
        digest = _sha256_text(text)
        sim = _simhash64(text)
        doc_id = f"repo::{rel.as_posix()}"
        stage_name = _sanitize_id(rel.as_posix()) + ".txt"
        target = stage_dir / "repo" / stage_name
        kind = _kind_for_path(path)
        _write_text_document(target, rel.as_posix(), kind, text)
        out.append(
            StagedRecord(
                doc_id=doc_id,
                source_path=rel.as_posix(),
                kind=kind,
                sha256=digest,
                bytes=len(raw),
                chars=len(text),
                staged_path=str(target),
                simhash=f"{sim:016x}",
            )
        )
    return out


@eidosian()
def stage_memory_and_kb_documents(
    repo_root: Path,
    stage_dir: Path,
    max_chars_per_doc: int,
) -> list[StagedRecord]:
    out: list[StagedRecord] = []
    for mem in extract_memory_records(repo_root / "memory_data.json"):
        content = mem["content"][:max_chars_per_doc]
        doc_id = f"memory::{mem['id']}"
        digest = _sha256_text(content)
        sim = _simhash64(content)
        target = stage_dir / "memory" / (_sanitize_id(mem["id"]) + ".txt")
        _write_text_document(target, mem["id"], "memory", content)
        out.append(
            StagedRecord(
                doc_id=doc_id,
                source_path=mem["id"],
                kind="memory",
                sha256=digest,
                bytes=len(content.encode("utf-8")),
                chars=len(content),
                staged_path=str(target),
                simhash=f"{sim:016x}",
            )
        )

    for node in extract_kb_records(repo_root / "data" / "kb.json"):
        content = node["content"][:max_chars_per_doc]
        doc_id = f"kb::{node['id']}"
        digest = _sha256_text(content)
        sim = _simhash64(content)
        target = stage_dir / "knowledge" / (_sanitize_id(node["id"]) + ".txt")
        _write_text_document(target, node["id"], "knowledge", content)
        out.append(
            StagedRecord(
                doc_id=doc_id,
                source_path=node["id"],
                kind="knowledge",
                sha256=digest,
                bytes=len(content.encode("utf-8")),
                chars=len(content),
                staged_path=str(target),
                simhash=f"{sim:016x}",
            )
        )
    return out


@eidosian()
def run_code_analysis(
    repo_root: Path,
    report_dir: Path,
    max_files: int | None = None,
) -> dict[str, Any]:
    db_path = repo_root / "data" / "code_forge" / "library.sqlite"
    runs_dir = repo_root / "data" / "code_forge" / "ingestion_runs"
    db = CodeLibraryDB(db_path)
    runner = IngestionRunner(db=db, runs_dir=runs_dir)
    stats = runner.ingest_path(
        repo_root,
        mode="analysis",
        max_files=max_files,
        progress_every=200,
        extensions=[".py"],
    )

    duplicate_groups = db.list_duplicate_units(min_occurrences=2, limit_groups=200)
    total_units = db.count_units()
    by_type = db.count_units_by_type()
    module_samples = [u for u in db.iter_units(limit=2000) if u.get("unit_type") == "module"][:12]
    traces = [
        db.trace_contains(str(sample["id"]), max_depth=2, max_nodes=120)
        for sample in module_samples
    ]

    code_report = {
        "generated_at": _now_utc(),
        "db_path": str(db_path),
        "run_stats": asdict(stats),
        "total_units": total_units,
        "units_by_type": by_type,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "trace_samples": traces,
    }
    report_path = report_dir / "code_analysis_report.json"
    report_path.write_text(json.dumps(code_report, indent=2) + "\n", encoding="utf-8")
    return code_report


def _pick_sweep_model(repo_root: Path, fallback: Path) -> Path:
    selection = repo_root / "reports" / "graphrag_sweep" / "model_selection_latest.json"
    if selection.exists():
        try:
            payload = json.loads(selection.read_text(encoding="utf-8"))
            model = (((payload.get("winner") or {}).get("model_path")) or "").strip()
            if model:
                candidate = (repo_root / model).resolve() if not Path(model).is_absolute() else Path(model)
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return fallback


def _wait_for_http(url: str, timeout_s: float = 60.0) -> None:
    import urllib.request

    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}")


def _llama_env(llama_server_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = str(llama_server_bin.resolve().parent)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld}" if ld else bin_dir
    return env


def _start_server(model_path: Path, port: int, embedding: bool = False) -> tuple[subprocess.Popen[Any], Any]:
    llama_server_bin = FORGE_ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
    if not llama_server_bin.exists():
        raise FileNotFoundError(f"Missing llama-server: {llama_server_bin}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    logs_dir = FORGE_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = (logs_dir / f"living_pipeline_server_{port}.log").open("a")
    cmd = [
        str(llama_server_bin),
        "-m",
        str(model_path),
        "--port",
        str(port),
        "--ctx-size",
        os.environ.get("EIDOS_LLAMA_CTX_SIZE", "4096"),
        "--temp",
        os.environ.get("EIDOS_LLAMA_TEMPERATURE", "0"),
        "--n-gpu-layers",
        os.environ.get("EIDOS_LLAMA_GPU_LAYERS", "0"),
    ]
    if embedding:
        cmd.extend(["--embedding", "--pooling", "mean"])
    else:
        cmd.extend(["--parallel", os.environ.get("EIDOS_LLAMA_PARALLEL", "1")])

    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=_llama_env(llama_server_bin))
    _wait_for_http(f"http://127.0.0.1:{port}/health", timeout_s=90.0)
    return proc, log_file


def _stop_servers(runtimes: list[tuple[subprocess.Popen[Any], Any]]) -> None:
    for proc, _log in runtimes:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc, log in runtimes:
        try:
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            log.close()
        except Exception:
            pass


def _write_workspace_settings(workspace_root: Path, llm_port: int, embed_port: int) -> None:
    prompts = workspace_root / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    text_prompt = prompts / "community_report_text.txt"
    graph_prompt = prompts / "community_report_graph.txt"
    text_prompt.write_text(TEXT_PROMPT, encoding="utf-8")
    graph_prompt.write_text(GRAPH_PROMPT, encoding="utf-8")

    settings = {
        "async_mode": "asyncio",
        "concurrent_requests": 1,
        "chunking": {"type": "tokens", "size": 300, "overlap": 100, "encoding_model": "o200k_base"},
        "input": {"type": "text", "encoding": "utf-8", "file_pattern": ".*\\.txt$$"},
        "input_storage": {"type": "file", "base_dir": "input"},
        "output_storage": {"type": "file", "base_dir": "output"},
        "update_output_storage": {"type": "file", "base_dir": "update_output"},
        "reporting": {"type": "file", "base_dir": "logs"},
        "cache": {"type": "json", "storage": {"type": "file", "base_dir": "cache"}},
        "completion_models": {
            "default_completion_model": {
                "type": "litellm",
                "model_provider": "openai",
                "model": "living-local-completion",
                "auth_method": "api_key",
                "api_key": "sk-no-key-required",
                "api_base": f"http://127.0.0.1:{llm_port}/v1",
                "call_args": {"temperature": 0.0, "max_tokens": 512, "max_completion_tokens": 512},
            }
        },
        "embedding_models": {
            "default_embedding_model": {
                "type": "litellm",
                "model_provider": "openai",
                "model": "living-local-embed",
                "auth_method": "api_key",
                "api_key": "sk-no-key-required",
                "api_base": f"http://127.0.0.1:{embed_port}/v1",
                "call_args": {"encoding_format": "float", "user": "graphrag"},
            }
        },
        "embed_text": {"embedding_model_id": "default_embedding_model"},
        "extract_graph": {"completion_model_id": "default_completion_model"},
        "summarize_descriptions": {"completion_model_id": "default_completion_model"},
        "community_reports": {
            "completion_model_id": "default_completion_model",
            "text_prompt": str(text_prompt.resolve()),
            "graph_prompt": str(graph_prompt.resolve()),
            "max_input_length": 4000,
            "max_length": 500,
        },
        "global_search": {"completion_model_id": "default_completion_model"},
        "local_search": {
            "completion_model_id": "default_completion_model",
            "embedding_model_id": "default_embedding_model",
        },
    }
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")


def _run_graphrag_index(workspace_root: Path, method: str) -> None:
    cmd = [sys.executable, "-m", "graphrag", "index", "--root", str(workspace_root), "--method", method]
    subprocess.run(cmd, check=True)


def _run_graphrag_query(workspace_root: Path, query: str, method: str = "global") -> str:
    cmd = [
        sys.executable,
        "-m",
        "graphrag",
        "query",
        "--root",
        str(workspace_root),
        "--method",
        method,
        "--response-type",
        "Concise Paragraph",
        query,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "graphrag query failed")
    return (proc.stdout or "").strip()


@eidosian()
def compare_with_previous_run(run_root: Path, records: list[StagedRecord]) -> dict[str, Any]:
    current = {r.source_path: r.sha256 for r in records}
    parent = run_root.parent
    prior_runs = sorted([p for p in parent.glob("*") if p.is_dir() and p.name != run_root.name])
    previous_manifest: dict[str, str] = {}
    if prior_runs:
        candidate = prior_runs[-1] / "manifest.json"
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                previous_manifest = dict(payload.get("source_hashes", {}))
            except Exception:
                previous_manifest = {}

    added = sorted([k for k in current.keys() if k not in previous_manifest])
    removed = sorted([k for k in previous_manifest.keys() if k not in current])
    changed = sorted(
        [k for k in current.keys() if k in previous_manifest and previous_manifest[k] != current[k]]
    )
    return {
        "previous_run": prior_runs[-1].name if prior_runs else None,
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added_sample": added[:50],
        "removed_sample": removed[:50],
        "changed_sample": changed[:50],
    }


@eidosian()
def run_pipeline(
    repo_root: Path,
    output_root: Path,
    workspace_root: Path,
    max_file_bytes: int,
    max_chars_per_doc: int,
    code_max_files: int | None,
    run_graphrag: bool,
    queries: list[str],
    method: str,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = output_root / run_id
    stage_dir = run_root / "staged_input"
    run_root.mkdir(parents=True, exist_ok=True)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    code_report = run_code_analysis(repo_root, run_root, max_files=code_max_files)
    repo_records = stage_repo_text_documents(repo_root, stage_dir, max_file_bytes, max_chars_per_doc)
    memory_records = stage_memory_and_kb_documents(repo_root, stage_dir, max_chars_per_doc)
    records = repo_records + memory_records

    exact_duplicates = group_exact_duplicates(records)
    near_duplicates = detect_near_duplicates(records)
    drift = compare_with_previous_run(run_root, records)

    workspace_input = workspace_root / "input"
    if workspace_input.exists():
        shutil.rmtree(workspace_input)
    workspace_input.mkdir(parents=True, exist_ok=True)
    for rec in records:
        src = Path(rec.staged_path)
        dest = workspace_input / src.relative_to(stage_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    graphrag_result: dict[str, Any] = {"indexed": False, "queries": []}
    if run_graphrag:
        llm_port = get_service_port("graphrag_llm", default=8081, env_keys=("EIDOS_GRAPHRAG_LLM_PORT",))
        embed_port = get_service_port("graphrag_embedding", default=8082, env_keys=("EIDOS_GRAPHRAG_EMBED_PORT",))
        llm_model = _pick_sweep_model(repo_root, repo_root / "models" / "Qwen2.5-0.5B-Instruct-Q8_0.gguf")
        embed_model = repo_root / "models" / "nomic-embed-text-v1.5.Q4_K_M.gguf"
        _write_workspace_settings(workspace_root, llm_port=llm_port, embed_port=embed_port)
        runtimes: list[tuple[subprocess.Popen[Any], Any]] = []
        try:
            runtimes.append(_start_server(llm_model, llm_port, embedding=False))
            runtimes.append(_start_server(embed_model, embed_port, embedding=True))
            _run_graphrag_index(workspace_root, method=method)
            graphrag_result["indexed"] = True
            graphrag_result["llm_model"] = str(llm_model)
            graphrag_result["embed_model"] = str(embed_model)
            for query in queries:
                answer = _run_graphrag_query(workspace_root, query, method="global")
                graphrag_result["queries"].append({"query": query, "answer": answer})
        finally:
            _stop_servers(runtimes)

    manifest = {
        "contract": "living_knowledge.pipeline.v1",
        "generated_at": _now_utc(),
        "run_id": run_id,
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "workspace_root": str(workspace_root),
        "records_total": len(records),
        "records_by_kind": dict(Counter(r.kind for r in records)),
        "exact_duplicate_groups": len(exact_duplicates),
        "near_duplicate_pairs": len(near_duplicates),
        "drift": drift,
        "code_analysis": {
            "run_id": code_report.get("run_stats", {}).get("run_id"),
            "files_processed": code_report.get("run_stats", {}).get("files_processed"),
            "units_created": code_report.get("run_stats", {}).get("units_created"),
            "total_units": code_report.get("total_units"),
            "duplicate_group_count": code_report.get("duplicate_group_count"),
        },
        "graphrag": graphrag_result,
        "source_hashes": {r.source_path: r.sha256 for r in records},
    }

    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_root / "records.jsonl").write_text(
        "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    (run_root / "duplicates_exact.json").write_text(json.dumps(exact_duplicates, indent=2) + "\n", encoding="utf-8")
    (run_root / "duplicates_near.json").write_text(json.dumps(near_duplicates, indent=2) + "\n", encoding="utf-8")
    (run_root / "drift.json").write_text(json.dumps(drift, indent=2) + "\n", encoding="utf-8")
    (output_root / "latest_run").write_text(run_id + "\n", encoding="utf-8")
    return manifest


@eidosian()
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a living knowledge/code corpus and optional GraphRAG index.")
    parser.add_argument("--repo-root", default=str(FORGE_ROOT), help="Forge repository root.")
    parser.add_argument("--output-root", default=str(FORGE_ROOT / "reports" / "living_knowledge"), help="Pipeline reports output root.")
    parser.add_argument("--workspace-root", default=str(FORGE_ROOT / "data" / "living_knowledge" / "workspace"), help="GraphRAG workspace root.")
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000, help="Skip tracked files larger than this size.")
    parser.add_argument("--max-chars-per-doc", type=int, default=20_000, help="Max characters per staged document.")
    parser.add_argument("--code-max-files", type=int, default=None, help="Optional max Python files for code analysis ingest.")
    parser.add_argument("--run-graphrag", action="store_true", help="Run GraphRAG index and queries after staging corpus.")
    parser.add_argument("--query", action="append", default=[], help="GraphRAG global query (repeatable).")
    parser.add_argument("--method", choices=["fast", "standard"], default="fast", help="GraphRAG index method.")
    return parser.parse_args()


@eidosian()
def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    manifest = run_pipeline(
        repo_root=repo_root,
        output_root=output_root,
        workspace_root=workspace_root,
        max_file_bytes=int(args.max_file_bytes),
        max_chars_per_doc=int(args.max_chars_per_doc),
        code_max_files=args.code_max_files,
        run_graphrag=bool(args.run_graphrag),
        queries=list(args.query or []),
        method=str(args.method),
    )
    print(json.dumps({"ok": True, "run_id": manifest["run_id"], "records_total": manifest["records_total"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
