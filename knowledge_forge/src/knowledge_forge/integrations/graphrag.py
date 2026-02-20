from eidosian_core import eidosian

"""
Integration with GraphRAG.
"""
import subprocess
from pathlib import Path
from typing import Any, Dict, List


class GraphRAGIntegration:
    """
    Bridge between KnowledgeForge and the external GraphRAG tool.
    """

    def __init__(self, graphrag_root: Path):
        self.root = graphrag_root

    @eidosian()
    def run_incremental_index(self, scan_roots: List[Path]) -> Dict[str, Any]:
        """
        Trigger an incremental index run.
        """
        cmd = ["python", "-m", "graphrag.index", "--root", str(self.root)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "command": cmd,
            "scan_roots": [str(r) for r in scan_roots],
            "success": res.returncode == 0,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }

    @eidosian()
    def global_query(self, query: str) -> Dict[str, Any]:
        """Run a global query against the index."""
        # python -m graphrag.query --root ./ragtest --method global "What are the top themes in this story?"
        cmd = ["python", "-m", "graphrag.query", "--root", str(self.root), "--method", "global", query]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "command": cmd,
            "success": res.returncode == 0,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }

    @eidosian()
    def local_query(self, query: str) -> Dict[str, Any]:
        """Run a local query against the index."""
        cmd = ["python", "-m", "graphrag.query", "--root", str(self.root), "--method", "local", query]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "command": cmd,
            "success": res.returncode == 0,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
