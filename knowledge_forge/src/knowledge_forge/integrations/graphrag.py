from eidosian_core import eidosian

"""
Integration with GraphRAG.
"""
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from eidosian_vector import build_default_embedder


def _forge_root() -> Path:
    raw = os.environ.get("EIDOS_FORGE_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


class GraphRAGIntegration:
    """
    Bridge between KnowledgeForge and the external GraphRAG tool.
    """

    def __init__(
        self,
        graphrag_root: Path,
        *,
        bridge: Any = None,
        memory_dir: Optional[Path] = None,
        kb_path: Optional[Path] = None,
    ):
        self.root = graphrag_root
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
        Trigger an incremental index run.
        """
        result = self._run_graphrag("index", "--root", str(self.root))
        result.update({"scan_roots": [str(r) for r in scan_roots]})
        return result

    @eidosian()
    def global_query(self, query: str) -> Dict[str, Any]:
        """Run a global query against the index."""
        return self._query_with_fallback(query, method="global")

    @eidosian()
    def local_query(self, query: str) -> Dict[str, Any]:
        """Run a local query against the index."""
        return self._query_with_fallback(query, method="local")
