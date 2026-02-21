from eidosian_core import eidosian

"""
Integration with GraphRAG.
"""
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List


class GraphRAGIntegration:
    """
    Bridge between KnowledgeForge and the external GraphRAG tool.
    """

    def __init__(self, graphrag_root: Path):
        self.root = graphrag_root
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
            "attempted_commands": [primary_cmd]
            + ([legacy_cmd] if fallback_used and legacy_cmd is not None else []),
            "fallback_used": fallback_used,
            "success": res.returncode == 0,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": int(res.returncode),
            "diagnostics": diagnostics,
        }

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
        return self._run_graphrag(
            "query", "--root", str(self.root), "--method", "global", query
        )

    @eidosian()
    def local_query(self, query: str) -> Dict[str, Any]:
        """Run a local query against the index."""
        return self._run_graphrag(
            "query", "--root", str(self.root), "--method", "local", query
        )
