"""
Integration with GraphRAG.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from eidosian_core import eidosian
from eidosian_vector import build_default_embedder

INDEXABLE_SUFFIXES = {
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

DEFAULT_CHUNK_CHARS = 1200
DEFAULT_CHUNK_OVERLAP = 200
MAX_NATIVE_REPORTS = 8


def _forge_root() -> Path:
    raw = os.environ.get("EIDOS_FORGE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


class GraphRAGIntegration:
    """
    Bridge between KnowledgeForge and the external GraphRAG tool.

    External GraphRAG remains supported when installed, but a native local
    vector+graph indexer now keeps the shared Eidosian knowledge substrate
    current even when GraphRAG itself is absent.
    """

    def __init__(
        self,
        graphrag_root: Path,
        *,
        bridge: Any = None,
        memory_dir: Optional[Path] = None,
        kb_path: Optional[Path] = None,
        word_graph_path: Optional[Path] = None,
        native_state_path: Optional[Path] = None,
    ):
        self.root = Path(graphrag_root).expanduser().resolve()
        self._bridge = bridge
        forge_root = _forge_root()
        self.memory_dir = (
            Path(memory_dir).expanduser().resolve()
            if memory_dir is not None
            else Path(os.environ.get("EIDOS_MEMORY_DIR", str(forge_root / "data" / "memory"))).expanduser().resolve()
        )
        self.kb_path = (
            Path(kb_path).expanduser().resolve()
            if kb_path is not None
            else Path(os.environ.get("EIDOS_KNOWLEDGE_PATH", str(forge_root / "data" / "kb.json")))
            .expanduser()
            .resolve()
        )
        self.word_graph_path = (
            Path(word_graph_path).expanduser().resolve()
            if word_graph_path is not None
            else forge_root / "data" / "eidos_semantic_graph.json"
        )
        self.native_state_path = (
            Path(native_state_path).expanduser().resolve()
            if native_state_path is not None
            else self.root / "native_graphrag_state.json"
        )
        timeout_raw = os.environ.get("EIDOS_GRAPHRAG_TIMEOUT_SEC", "900")
        try:
            self.timeout_seconds = max(30, int(timeout_raw))
        except Exception:
            self.timeout_seconds = 900

    @staticmethod
    def _is_legacy_fallback_candidate(stderr: str) -> bool:
        message = (stderr or "").lower()
        return any(
            token in message
            for token in (
                "no module named graphrag.__main__",
                "no such command",
                "usage: graphrag.index",
                "usage: graphrag.query",
            )
        )

    def _run_graphrag(self, *args: str) -> Dict[str, Any]:
        """
        Execute GraphRAG with a compatibility fallback across CLI shapes.

        Preferred (new): ``python -m graphrag <subcommand> ...``
        Fallback (legacy): ``python -m graphrag.<subcommand> ...``
        """
        python_bin = str(sys.executable or "python")
        primary_cmd = [python_bin, "-m", "graphrag", *args]
        legacy_cmd = None
        if args:
            legacy_cmd = [python_bin, "-m", f"graphrag.{args[0]}", *args[1:]]

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                primary_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "command": primary_cmd,
                "attempted_commands": [primary_cmd],
                "fallback_used": False,
                "success": False,
                "stdout": exc.stdout or "",
                "stderr": f"GraphRAG primary command timed out after {self.timeout_seconds}s",
                "returncode": 124,
                "diagnostics": {"timeout_seconds": self.timeout_seconds},
            }
        used_cmd = primary_cmd
        fallback_used = False
        diagnostics: Dict[str, Any] = {}
        if res.returncode != 0 and legacy_cmd is not None:
            if self._is_legacy_fallback_candidate(res.stderr or ""):
                try:
                    legacy = subprocess.run(
                        legacy_cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                except subprocess.TimeoutExpired as exc:
                    return {
                        "command": primary_cmd,
                        "attempted_commands": [primary_cmd, legacy_cmd],
                        "fallback_used": True,
                        "success": False,
                        "stdout": exc.stdout or "",
                        "stderr": f"GraphRAG legacy fallback timed out after {self.timeout_seconds}s",
                        "returncode": 124,
                        "diagnostics": {"timeout_seconds": self.timeout_seconds},
                    }
                fallback_used = True
                diagnostics["fallback_stderr"] = legacy.stderr
                diagnostics["fallback_stdout"] = legacy.stdout
                diagnostics["fallback_returncode"] = int(legacy.returncode)
                if legacy.returncode == 0:
                    res = legacy
                    used_cmd = legacy_cmd

        return {
            "command": used_cmd,
            "attempted_commands": [primary_cmd] + ([legacy_cmd] if fallback_used and legacy_cmd is not None else []),
            "fallback_used": fallback_used,
            "success": res.returncode == 0,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": int(res.returncode),
            "diagnostics": diagnostics,
        }

    def _load_bridge(self):
        if self._bridge is not None:
            return self._bridge
        try:
            from knowledge_forge.core.bridge import KnowledgeMemoryBridge

            self._bridge = KnowledgeMemoryBridge(
                memory_dir=self.memory_dir,
                kb_path=self.kb_path,
                embedder=build_default_embedder(),
            )
        except Exception:
            self._bridge = None
        return self._bridge

    def _load_native_state(self) -> Dict[str, Any]:
        if not self.native_state_path.exists():
            return {"files": {}, "word_graph": {}}
        try:
            payload = json.loads(self.native_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"files": {}, "word_graph": {}}
        if not isinstance(payload, dict):
            return {"files": {}, "word_graph": {}}
        payload.setdefault("files", {})
        payload.setdefault("word_graph", {})
        return payload

    def _save_native_state(self, state: Dict[str, Any]) -> None:
        self.native_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.native_state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def _iter_indexable_files(self, scan_roots: Iterable[Path]) -> Iterable[Path]:
        seen: set[Path] = set()
        for root in scan_roots:
            candidate = Path(root).expanduser().resolve()
            if not candidate.exists():
                continue
            if candidate.is_file():
                if self._is_indexable_path(candidate) and candidate not in seen:
                    seen.add(candidate)
                    yield candidate
                continue
            for path in candidate.rglob("*"):
                if path in seen:
                    continue
                if not path.is_file():
                    continue
                if self._is_indexable_path(path):
                    seen.add(path)
                    yield path

    def _is_indexable_path(self, path: Path) -> bool:
        if path.suffix.lower() not in INDEXABLE_SUFFIXES:
            return False
        return not any(segment in EXCLUDE_SEGMENTS for segment in path.parts)

    def _read_text_file(self, path: Path) -> str:
        try:
            data = path.read_bytes()
        except Exception:
            return ""
        if b"\x00" in data:
            return ""
        return data.decode("utf-8", errors="ignore")

    def _chunk_text(self, text: str) -> List[str]:
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        if len(cleaned) <= DEFAULT_CHUNK_CHARS:
            return [cleaned]
        chunks: list[str] = []
        start = 0
        step = max(1, DEFAULT_CHUNK_CHARS - DEFAULT_CHUNK_OVERLAP)
        while start < len(cleaned):
            end = min(len(cleaned), start + DEFAULT_CHUNK_CHARS)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start += step
        return chunks

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def _delete_recorded_nodes(self, knowledge: Any, node_ids: Iterable[str]) -> int:
        deleted = 0
        for node_id in node_ids:
            try:
                if knowledge.delete_node(str(node_id)):
                    deleted += 1
            except Exception:
                continue
        return deleted

    def _ingest_word_graph(self, knowledge: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "present": self.word_graph_path.exists(),
            "changed": False,
            "term_nodes": 0,
            "relationships": 0,
            "deleted_nodes": 0,
        }
        if not self.word_graph_path.exists():
            state["word_graph"] = {}
            return summary

        payload = json.loads(self.word_graph_path.read_text(encoding="utf-8"))
        digest = self._sha256_file(self.word_graph_path)
        prior = dict(state.get("word_graph") or {})
        if prior.get("sha256") == digest:
            summary["term_nodes"] = int(prior.get("term_nodes", 0))
            summary["relationships"] = int(prior.get("relationships", 0))
            return summary

        prior_nodes = list(prior.get("node_ids") or [])
        if prior_nodes:
            summary["deleted_nodes"] = self._delete_recorded_nodes(knowledge, prior_nodes)

        node_id_map: dict[str, str] = {}
        created_node_ids: list[str] = []
        for node in payload.get("nodes", []):
            term = str(node.get("id") or "").strip()
            if not term:
                continue
            attributes = {k: v for k, v in node.items() if k != "id"}
            lines = [f"Word Forge term: {term}"]
            if attributes.get("definition"):
                lines.append(f"Definition: {attributes['definition']}")
            if attributes.get("pos"):
                lines.append(f"Part of speech: {attributes['pos']}")
            if attributes.get("source"):
                lines.append(f"Source: {attributes['source']}")
            content = "\n".join(lines)
            metadata = {
                "source": "word_forge",
                "kind": "lexicon_term",
                "term": term,
                "attributes": attributes,
                "source_paths": [str(self.word_graph_path)],
            }
            k_node = knowledge.add_knowledge(
                content=content,
                concepts=[term],
                tags=["word_forge", "lexicon", "term"],
                metadata=metadata,
            )
            node_id_map[term] = k_node.id
            created_node_ids.append(k_node.id)

        relationships = 0
        for edge in payload.get("edges", []):
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            if not source or not target:
                continue
            source_id = node_id_map.get(source)
            target_id = node_id_map.get(target)
            if not source_id or not target_id:
                continue
            knowledge.link_nodes(source_id, target_id)
            relationships += 1

        state["word_graph"] = {
            "sha256": digest,
            "node_ids": created_node_ids,
            "term_nodes": len(created_node_ids),
            "relationships": relationships,
            "updated_at": self._timestamp(),
            "path": str(self.word_graph_path),
        }
        summary["changed"] = True
        summary["term_nodes"] = len(created_node_ids)
        summary["relationships"] = relationships
        return summary

    def _timestamp(self) -> str:
        try:
            from datetime import datetime, timezone

            return datetime.now(timezone.utc).isoformat()
        except Exception:
            return ""

    def _document_kind(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".py", ".sh", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp"}:
            return "code"
        if suffix in {".md", ".rst", ".txt"}:
            return "docs"
        return "data"

    def _default_code_forge_artifact_paths(self) -> List[Path]:
        forge_root = _forge_root()
        candidates = [
            forge_root / "data" / "code_forge",
            forge_root / "reports",
        ]
        return [path.resolve() for path in candidates if path.exists()]

    def _iter_code_forge_artifact_files(self, scan_roots: List[Path]) -> Iterable[Path]:
        wanted = {
            "provenance_registry.json",
            "provenance_links.json",
            "drift_report.json",
            "triage.json",
            "triage_audit.json",
        }
        seen: set[Path] = set()
        for root in list(scan_roots) + self._default_code_forge_artifact_paths():
            base = Path(root).expanduser().resolve()
            if not base.exists():
                continue
            if base.is_file():
                if base.name in wanted or ("code_forge" in base.name and base.suffix.lower() == ".json"):
                    if base not in seen:
                        seen.add(base)
                        yield base
                continue
            for path in base.rglob("*"):
                if path in seen or not path.is_file():
                    continue
                if path.name in wanted or ("code_forge" in path.name and path.suffix.lower() == ".json"):
                    seen.add(path)
                    yield path

    def _summarize_json_payload(self, payload: Dict[str, Any], *, max_items: int = 6) -> str:
        lines: list[str] = []
        for key, value in payload.items():
            if len(lines) >= max_items:
                break
            if isinstance(value, dict):
                sub_keys = list(value.keys())[:4]
                lines.append(f"{key}: object[{', '.join(str(k) for k in sub_keys)}]")
            elif isinstance(value, list):
                lines.append(f"{key}: list[{len(value)}]")
            elif isinstance(value, (str, int, float, bool)) or value is None:
                text = str(value)
                if len(text) > 140:
                    text = text[:137] + "..."
                lines.append(f"{key}: {text}")
        return "\n".join(lines).strip()

    def _artifact_kind(self, path: Path, payload: Dict[str, Any]) -> str:
        if path.name == "provenance_registry.json":
            return "code_forge_provenance_registry"
        if path.name == "provenance_links.json":
            return "code_forge_provenance_links"
        if path.name == "drift_report.json":
            return "code_forge_drift"
        if path.name == "triage_audit.json":
            return "code_forge_triage_audit"
        if path.name == "triage.json":
            return "code_forge_triage"
        if "benchmark" in path.name:
            return "code_forge_benchmark"
        return str(payload.get("schema_version") or payload.get("stage") or "code_forge_artifact")

    def _ingest_code_forge_artifacts(
        self, knowledge: Any, state: Dict[str, Any], scan_roots: List[Path]
    ) -> Dict[str, Any]:
        artifact_state = dict(state.get("code_forge_artifacts") or {})
        active_paths: set[str] = set()
        artifacts_seen = 0
        artifacts_indexed = 0
        artifacts_unchanged = 0
        artifacts_removed = 0
        links_created = 0
        deleted_nodes = 0

        for path in self._iter_code_forge_artifact_files(scan_roots):
            artifacts_seen += 1
            source_key = str(path)
            active_paths.add(source_key)
            digest = self._sha256_file(path)
            prior = dict(artifact_state.get(source_key) or {})
            if prior.get("sha256") == digest:
                artifacts_unchanged += 1
                continue

            if prior:
                deleted_nodes += self._delete_recorded_nodes(
                    knowledge,
                    [str(prior.get("root_node_id") or "")] + list(prior.get("node_ids") or []),
                )

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                artifact_state.pop(source_key, None)
                continue
            if not isinstance(payload, dict):
                artifact_state.pop(source_key, None)
                continue

            rel_path = os.path.relpath(path, _forge_root())
            kind = self._artifact_kind(path, payload)
            title = str(payload.get("provenance_id") or payload.get("registry_id") or payload.get("stage") or path.stem)
            root_node = knowledge.add_knowledge(
                content=f"Code Forge artifact: {rel_path}\nKind: {kind}\nTitle: {title}",
                concepts=[path.stem, "code_forge_artifact"],
                tags=["code_forge", "artifact", kind],
                metadata={
                    "source": "code_forge",
                    "kind": kind,
                    "artifact_path": rel_path,
                    "source_paths": [str(path)],
                    "sha256": digest,
                },
            )

            detail_ids: list[str] = []
            summary_text = self._summarize_json_payload(payload)
            if summary_text:
                summary_node = knowledge.add_knowledge(
                    content=f"Artifact summary: {rel_path}\n{summary_text}",
                    concepts=[path.stem],
                    tags=["code_forge", "artifact_summary", kind],
                    metadata={
                        "source": "code_forge",
                        "kind": "artifact_summary",
                        "artifact_path": rel_path,
                        "source_paths": [str(path)],
                        "sha256": digest,
                    },
                )
                knowledge.link_nodes(root_node.id, summary_node.id)
                detail_ids.append(summary_node.id)

            unit_links = []
            links_payload = payload.get("links")
            if isinstance(links_payload, dict):
                candidate_links = links_payload.get("unit_links")
                if isinstance(candidate_links, list):
                    unit_links = [row for row in candidate_links if isinstance(row, dict)]
            if not unit_links and path.name == "provenance_links.json":
                gr_docs = (
                    ((payload.get("graphrag_links") or {}).get("documents"))
                    if isinstance(payload.get("graphrag_links"), dict)
                    else []
                )
                if isinstance(gr_docs, list):
                    unit_links = [row for row in gr_docs if isinstance(row, dict)]

            for row in unit_links[:200]:
                knowledge_node_id = str(row.get("knowledge_node_id") or row.get("node_id") or "").strip()
                if knowledge_node_id and knowledge_node_id in knowledge.nodes:
                    knowledge.link_nodes(root_node.id, knowledge_node_id)
                    links_created += 1
                else:
                    unit_id = str(row.get("unit_id") or "").strip()
                    qualified_name = str(row.get("qualified_name") or unit_id or "").strip()
                    if not qualified_name:
                        continue
                    detail = knowledge.add_knowledge(
                        content=f"Code Forge unit linkage: {qualified_name}\nArtifact: {rel_path}",
                        concepts=[qualified_name],
                        tags=["code_forge", "unit_link", kind],
                        metadata={
                            "source": "code_forge",
                            "kind": "unit_link",
                            "unit_id": unit_id or None,
                            "qualified_name": qualified_name,
                            "artifact_path": rel_path,
                            "source_paths": [str(path)],
                        },
                    )
                    knowledge.link_nodes(root_node.id, detail.id)
                    detail_ids.append(detail.id)

            benchmark = payload.get("benchmark")
            if isinstance(benchmark, dict):
                lines = ["Benchmark summary:"]
                gate_pass = benchmark.get("gate_pass")
                if gate_pass is not None:
                    lines.append(f"gate_pass: {gate_pass}")
                if benchmark.get("search_p95_ms") is not None:
                    lines.append(f"search_p95_ms: {benchmark.get('search_p95_ms')}")
                if benchmark.get("graph_build_ms") is not None:
                    lines.append(f"graph_build_ms: {benchmark.get('graph_build_ms')}")
                bench_node = knowledge.add_knowledge(
                    content="\n".join(lines),
                    concepts=["benchmark", "code_forge"],
                    tags=["code_forge", "benchmark", kind],
                    metadata={
                        "source": "code_forge",
                        "kind": "benchmark_summary",
                        "artifact_path": rel_path,
                        "source_paths": [str(path)],
                    },
                )
                knowledge.link_nodes(root_node.id, bench_node.id)
                detail_ids.append(bench_node.id)

            drift = payload.get("drift")
            if isinstance(drift, dict) and drift.get("warning_count") is not None:
                drift_node = knowledge.add_knowledge(
                    content=f"Drift warnings: {drift.get('warning_count')}\nMax abs delta: {drift.get('max_abs_delta')}",
                    concepts=["drift", "code_forge"],
                    tags=["code_forge", "drift", kind],
                    metadata={
                        "source": "code_forge",
                        "kind": "drift_summary",
                        "artifact_path": rel_path,
                        "source_paths": [str(path)],
                    },
                )
                knowledge.link_nodes(root_node.id, drift_node.id)
                detail_ids.append(drift_node.id)

            artifact_state[source_key] = {
                "sha256": digest,
                "root_node_id": root_node.id,
                "node_ids": detail_ids,
                "kind": kind,
                "artifact_path": rel_path,
                "updated_at": self._timestamp(),
            }
            artifacts_indexed += 1

        for stale_path in sorted(set(artifact_state.keys()) - active_paths):
            prior = dict(artifact_state.get(stale_path) or {})
            deleted_nodes += self._delete_recorded_nodes(
                knowledge,
                [str(prior.get("root_node_id") or "")] + list(prior.get("node_ids") or []),
            )
            artifact_state.pop(stale_path, None)
            artifacts_removed += 1

        state["code_forge_artifacts"] = artifact_state
        return {
            "artifacts_seen": artifacts_seen,
            "artifacts_indexed": artifacts_indexed,
            "artifacts_unchanged": artifacts_unchanged,
            "artifacts_removed": artifacts_removed,
            "links_created": links_created,
            "deleted_nodes": deleted_nodes,
        }

    def _collect_report_candidate_ids(self, state: Dict[str, Any]) -> List[str]:
        node_ids: list[str] = []
        for payload in (state.get("files") or {}).values():
            if isinstance(payload, dict):
                node_ids.extend(
                    [str(payload.get("root_node_id") or "")] + [str(x) for x in payload.get("chunk_node_ids") or []]
                )
        word_graph = state.get("word_graph") or {}
        if isinstance(word_graph, dict):
            node_ids.extend(str(x) for x in word_graph.get("node_ids") or [])
        for payload in (state.get("code_forge_artifacts") or {}).values():
            if isinstance(payload, dict):
                node_ids.extend(
                    [str(payload.get("root_node_id") or "")] + [str(x) for x in payload.get("node_ids") or []]
                )
        return [node_id for node_id in dict.fromkeys(node_ids) if node_id]

    def _build_native_community_reports(self, knowledge: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        output_dir = self.root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate_ids = self._collect_report_candidate_ids(state)
        nodes = [knowledge.nodes[node_id] for node_id in candidate_ids if node_id in knowledge.nodes]

        buckets: dict[str, list[Any]] = {
            "documents": [node for node in nodes if "document_chunk" in node.tags or "document_root" in node.tags],
            "lexicon": [node for node in nodes if "word_forge" in node.tags or "lexicon" in node.tags],
            "code_forge": [node for node in nodes if "code_forge" in node.tags or "artifact" in node.tags],
        }
        tag_counts = Counter(
            tag
            for node in nodes
            for tag in node.tags
            if tag not in {"native_graphrag", "document_root", "document_chunk", "code_forge", "artifact", "word_forge"}
        )
        for tag, _count in tag_counts.most_common(MAX_NATIVE_REPORTS):
            group = [node for node in nodes if tag in node.tags]
            if len(group) >= 2:
                buckets.setdefault(tag, group)

        reports: list[dict[str, Any]] = []
        for community, group in buckets.items():
            if not group:
                continue
            unique_group = list(dict.fromkeys(group))
            sample_nodes = unique_group[:5]
            findings = []
            linked_neighbors = 0
            for node in sample_nodes:
                content = str(node.content).strip().replace("\n", " ")
                if len(content) > 180:
                    content = content[:177] + "..."
                findings.append(content)
                linked_neighbors += len(knowledge.get_related_nodes(node.id))
            rating = min(5, max(1, len(unique_group) // 2 + (1 if linked_neighbors else 0)))
            reports.append(
                {
                    "community": community,
                    "title": f"{community.replace('_', ' ').title()} Community",
                    "summary": f"{len(unique_group)} nodes with {linked_neighbors} linked neighbor references",
                    "rating": rating,
                    "rating_explanation": "Higher scores reflect denser linked context and broader coverage.",
                    "findings": findings,
                    "node_ids": [node.id for node in unique_group[:25]],
                }
            )

        reports = sorted(
            reports, key=lambda row: (int(row.get("rating", 0)), len(row.get("node_ids", []))), reverse=True
        )
        json_path = output_dir / "native_community_reports.json"
        md_path = output_dir / "native_community_reports.md"
        json_path.write_text(
            json.dumps({"generated_at": self._timestamp(), "reports": reports}, indent=2) + "\n", encoding="utf-8"
        )

        md_lines = ["# Native Community Reports", ""]
        for report in reports:
            md_lines.append(f"## {report['title']}")
            md_lines.append(f"- Community: {report['community']}")
            md_lines.append(f"- Rating: {report['rating']}")
            md_lines.append(f"- Summary: {report['summary']}")
            md_lines.append("- Findings:")
            for finding in report["findings"]:
                md_lines.append(f"  - {finding}")
            md_lines.append("")
        md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
        return {
            "count": len(reports),
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "top_community": reports[0]["community"] if reports else None,
        }

    def _load_native_reports_payload(self) -> Dict[str, Any]:
        path = self.root / "output" / "native_community_reports.json"
        if not path.exists():
            return {"generated_at": None, "reports": [], "path": str(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"generated_at": None, "reports": [], "path": str(path)}
        if not isinstance(payload, dict):
            return {"generated_at": None, "reports": [], "path": str(path)}
        payload["path"] = str(path)
        if not isinstance(payload.get("reports"), list):
            payload["reports"] = []
        return payload

    @eidosian()
    def native_report_summary(self, limit: int = 5) -> Dict[str, Any]:
        payload = self._load_native_reports_payload()
        reports = [row for row in payload.get("reports", []) if isinstance(row, dict)]
        max_reports = max(1, int(limit or 5))
        summaries: list[dict[str, Any]] = []
        for row in reports[:max_reports]:
            summaries.append(
                {
                    "community": row.get("community"),
                    "title": row.get("title"),
                    "rating": row.get("rating"),
                    "summary": row.get("summary"),
                    "finding_count": len(row.get("findings") or []),
                    "node_count": len(row.get("node_ids") or []),
                }
            )
        return {
            "generated_at": payload.get("generated_at"),
            "path": payload.get("path"),
            "count": len(reports),
            "reports": summaries,
        }

    @eidosian()
    def native_artifact_summary(self, limit: int = 10) -> Dict[str, Any]:
        state = self._load_native_state()
        artifacts = [value for value in (state.get("code_forge_artifacts") or {}).values() if isinstance(value, dict)]
        items = sorted(
            artifacts,
            key=lambda row: str(row.get("updated_at") or ""),
            reverse=True,
        )
        max_items = max(1, int(limit or 10))
        summary_items = [
            {
                "kind": row.get("kind"),
                "artifact_path": row.get("artifact_path"),
                "updated_at": row.get("updated_at"),
                "detail_node_count": len(row.get("node_ids") or []),
            }
            for row in items[:max_items]
        ]
        kind_counts = Counter(str(row.get("kind") or "unknown") for row in artifacts)
        return {
            "count": len(artifacts),
            "kinds": dict(sorted(kind_counts.items())),
            "items": summary_items,
            "state_path": str(self.native_state_path),
        }

    def _native_index(self, scan_roots: List[Path]) -> Dict[str, Any]:
        bridge = self._load_bridge()
        if bridge is None or getattr(bridge, "knowledge", None) is None:
            return {
                "success": False,
                "mode": "native_vector_graph",
                "stderr": "Knowledge bridge unavailable for native GraphRAG indexing",
                "scan_roots": [str(Path(root)) for root in scan_roots],
            }

        knowledge = bridge.knowledge
        state = self._load_native_state()
        file_state = dict(state.get("files") or {})
        active_paths: set[str] = set()
        files_seen = 0
        files_indexed = 0
        files_unchanged = 0
        files_removed = 0
        deleted_nodes = 0
        document_nodes = 0
        chunk_nodes = 0

        for path in self._iter_indexable_files(scan_roots):
            files_seen += 1
            source_key = str(path)
            active_paths.add(source_key)
            digest = self._sha256_file(path)
            prior = dict(file_state.get(source_key) or {})
            if prior.get("sha256") == digest:
                files_unchanged += 1
                continue

            if prior:
                deleted_nodes += self._delete_recorded_nodes(
                    knowledge,
                    [str(prior.get("root_node_id") or "")] + list(prior.get("chunk_node_ids") or []),
                )

            text = self._read_text_file(path)
            if not text.strip():
                file_state.pop(source_key, None)
                continue

            rel_path = os.path.relpath(path, _forge_root())
            doc_kind = self._document_kind(path)
            root_node = knowledge.add_knowledge(
                content=f"Document: {rel_path}",
                concepts=[path.stem],
                tags=["native_graphrag", "document_root", doc_kind],
                metadata={
                    "source": "native_graphrag",
                    "kind": "document_root",
                    "document_path": rel_path,
                    "source_paths": [str(path)],
                    "sha256": digest,
                },
            )
            document_nodes += 1

            chunk_ids: list[str] = []
            chunks = self._chunk_text(text)
            for index, chunk in enumerate(chunks, start=1):
                node = knowledge.add_knowledge(
                    content=f"Document: {rel_path}\nChunk {index}/{len(chunks)}\n\n{chunk}",
                    concepts=[path.stem],
                    tags=["native_graphrag", "document_chunk", doc_kind, path.suffix.lower().lstrip(".") or "text"],
                    metadata={
                        "source": "native_graphrag",
                        "kind": "document_chunk",
                        "document_path": rel_path,
                        "source_paths": [str(path)],
                        "chunk_index": index,
                        "chunk_total": len(chunks),
                        "sha256": digest,
                    },
                )
                knowledge.link_nodes(root_node.id, node.id)
                if chunk_ids:
                    knowledge.link_nodes(chunk_ids[-1], node.id)
                chunk_ids.append(node.id)
                chunk_nodes += 1

            file_state[source_key] = {
                "sha256": digest,
                "root_node_id": root_node.id,
                "chunk_node_ids": chunk_ids,
                "chunk_count": len(chunk_ids),
                "document_path": rel_path,
                "updated_at": self._timestamp(),
            }
            files_indexed += 1

        for stale_path in sorted(set(file_state.keys()) - active_paths):
            prior = dict(file_state.get(stale_path) or {})
            deleted_nodes += self._delete_recorded_nodes(
                knowledge,
                [str(prior.get("root_node_id") or "")] + list(prior.get("chunk_node_ids") or []),
            )
            file_state.pop(stale_path, None)
            files_removed += 1

        state["files"] = file_state
        word_forge = self._ingest_word_graph(knowledge, state)
        code_forge = self._ingest_code_forge_artifacts(knowledge, state, scan_roots)
        community_reports = self._build_native_community_reports(knowledge, state)
        self._save_native_state(state)
        knowledge.save()

        return {
            "success": True,
            "mode": "native_vector_graph",
            "scan_roots": [str(Path(root).expanduser().resolve()) for root in scan_roots],
            "state_path": str(self.native_state_path),
            "files_seen": files_seen,
            "files_indexed": files_indexed,
            "files_unchanged": files_unchanged,
            "files_removed": files_removed,
            "deleted_nodes": deleted_nodes,
            "document_nodes": document_nodes,
            "chunk_nodes": chunk_nodes,
            "word_forge": word_forge,
            "code_forge": code_forge,
            "community_reports": community_reports,
        }

    def _local_vector_graph_query(self, query: str, method: str) -> Dict[str, Any]:
        bridge = self._load_bridge()
        if bridge is None:
            return {
                "success": False,
                "mode": "local_vector_graph",
                "method": method,
                "query": query,
                "summary": "",
                "memory_context": [],
                "knowledge_context": [],
                "graph_neighbors": [],
                "graph_paths": [],
                "local_fallback": False,
            }

        context = bridge.get_memory_knowledge_context(query, max_results=8)
        knowledge_ids = [str(entry.get("id")) for entry in context.get("knowledge_context", []) if entry.get("id")]
        graph_neighbors: list[dict[str, Any]] = []
        graph_paths: list[list[str]] = []
        knowledge = getattr(bridge, "knowledge", None)
        if knowledge is not None:
            seen_neighbors: set[str] = set()
            for node_id in knowledge_ids[:3]:
                for node in knowledge.get_related_nodes(node_id)[:6]:
                    if node.id in seen_neighbors:
                        continue
                    seen_neighbors.add(node.id)
                    graph_neighbors.append(
                        {
                            "id": node.id,
                            "content": str(node.content)[:220],
                            "tags": sorted(list(node.tags)),
                        }
                    )
            if len(knowledge_ids) >= 2:
                for idx in range(len(knowledge_ids) - 1):
                    path = knowledge.find_path(knowledge_ids[idx], knowledge_ids[idx + 1])
                    if path:
                        graph_paths.append(path)

        summary_parts: list[str] = []
        if context.get("knowledge_context"):
            top = context["knowledge_context"][0]
            summary_parts.append(f"Top knowledge hit: {str(top.get('content') or '')[:160]}")
        if context.get("memory_context"):
            top = context["memory_context"][0]
            summary_parts.append(f"Top memory hit: {str(top.get('content') or '')[:160]}")
        if graph_neighbors:
            summary_parts.append(f"Linked graph neighbors: {len(graph_neighbors)}")
        if graph_paths:
            summary_parts.append(f"Graph paths discovered: {len(graph_paths)}")

        return {
            "success": bool(context.get("total_results") or graph_neighbors),
            "mode": "local_vector_graph",
            "method": method,
            "query": query,
            "summary": " | ".join(summary_parts),
            "memory_context": list(context.get("memory_context") or []),
            "knowledge_context": list(context.get("knowledge_context") or []),
            "graph_neighbors": graph_neighbors,
            "graph_paths": graph_paths,
            "local_fallback": True,
        }

    def _query_with_fallback(self, query: str, *, method: str) -> Dict[str, Any]:
        result = self._run_graphrag("query", "--root", str(self.root), "--method", method, query)
        if result.get("success"):
            return result
        local = self._local_vector_graph_query(query, method)
        if not local.get("success"):
            return result
        merged = dict(result)
        merged["success"] = True
        merged["fallback_used"] = True
        merged["local_fallback"] = True
        merged["mode"] = str(local.get("mode"))
        merged["method"] = method
        merged["query"] = query
        merged["summary"] = str(local.get("summary") or "")
        merged["memory_context"] = list(local.get("memory_context") or [])
        merged["knowledge_context"] = list(local.get("knowledge_context") or [])
        merged["graph_neighbors"] = list(local.get("graph_neighbors") or [])
        merged["graph_paths"] = list(local.get("graph_paths") or [])
        return merged

    @eidosian()
    def run_incremental_index(self, scan_roots: List[Path]) -> Dict[str, Any]:
        """
        Trigger an index run.

        Native indexing always updates the shared knowledge/vector substrate.
        External GraphRAG is treated as an optional augmentation layer.
        """
        scan_roots = [Path(root).expanduser().resolve() for root in scan_roots]
        native = self._native_index(scan_roots)
        external = self._run_graphrag("index", "--root", str(self.root))

        result = dict(native)
        result["scan_roots"] = [str(root) for root in scan_roots]
        result["external"] = external
        if external.get("success"):
            result["mode"] = "external+native_vector_graph"
            result["external_success"] = True
            result["command"] = external.get("command")
            result["attempted_commands"] = external.get("attempted_commands", [])
            result["fallback_used"] = bool(external.get("fallback_used"))
        else:
            result["external_success"] = False
            result["external_error"] = external.get("stderr", "")

        result["success"] = bool(native.get("success"))
        return result

    @eidosian()
    def global_query(self, query: str) -> Dict[str, Any]:
        """Run a global query against the index."""
        return self._query_with_fallback(query, method="global")

    @eidosian()
    def local_query(self, query: str) -> Dict[str, Any]:
        """Run a local query against the index."""
        return self._query_with_fallback(query, method="local")
