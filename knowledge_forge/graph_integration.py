"""
Integration between KnowledgeForge and GraphRAG.
Provides high-level indexing and search capabilities.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

class GraphRAGIntegration:
    """
    Handles communication with the GraphRAG indexing and query systems.
    """
    def __init__(self, graphrag_root: Path):
        self.root = graphrag_root
        self.index_script = Path(__file__).parent.parent / "scripts" / "graphrag_local_index.py"

    def run_incremental_index(self, scan_roots: List[Path], verbose: bool = False) -> Dict[str, Any]:
        """
        Trigger an incremental index run using graphrag_local_index.py.
        """
        cmd = [
            sys.executable,
            str(self.index_script),
            "--root", str(self.root),
        ]
        for sr in scan_roots:
            cmd.extend(["--scan-root", str(sr)])
            
        if verbose:
            cmd.append("--verbose")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": e.stderr,
                "stdout": e.stdout
            }

    def global_query(self, query: str) -> Dict[str, Any]:
        """
        Perform a global query using GraphRAG CLI.
        """
        cmd = [
            "python3", "-m", "graphrag.query.cli",
            "--root", str(self.root),
            "--method", "global",
            query
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {"success": True, "response": result.stdout}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": e.stderr}

    def local_query(self, query: str) -> Dict[str, Any]:
        """
        Perform a local query using GraphRAG CLI.
        """
        cmd = [
            "python3", "-m", "graphrag.query.cli",
            "--root", str(self.root),
            "--method", "local",
            query
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {"success": True, "response": result.stdout}
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": e.stderr}
