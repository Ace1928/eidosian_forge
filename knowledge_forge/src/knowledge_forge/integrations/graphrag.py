"""
Integration with GraphRAG.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
