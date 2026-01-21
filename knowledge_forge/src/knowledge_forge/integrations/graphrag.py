"""
Integration with GraphRAG.
"""
import subprocess
from pathlib import Path
from typing import List, Dict, Any

class GraphRAGIntegration:
    """
    Bridge between KnowledgeForge and the external GraphRAG tool.
    """
    def __init__(self, graphrag_root: Path):
        self.root = graphrag_root
        
    def run_incremental_index(self, scan_roots: List[Path]) -> Dict[str, Any]:
        """
        Trigger an incremental index run.
        """
        # This is a placeholder for the actual CLI call to graphrag
        # GraphRAG usually runs via `python -m graphrag.index`
        cmd = ["python", "-m", "graphrag.index", "--root", str(self.root)]
        
        # For now, just return the command we would run
        return {"command": cmd, "scan_roots": [str(r) for r in scan_roots], "success": True, "stdout": "Simulated indexing"}

    def global_query(self, query: str) -> Dict[str, Any]:
        """Run a global query against the index."""
        # python -m graphrag.query --root ./ragtest --method global "What are the top themes in this story?"
        cmd = ["python", "-m", "graphrag.query", "--root", str(self.root), "--method", "global", query]
        return {"command": cmd, "success": True, "response": "Simulated global answer"}

    def local_query(self, query: str) -> Dict[str, Any]:
        """Run a local query against the index."""
        cmd = ["python", "-m", "graphrag.query", "--root", str(self.root), "--method", "local", query]
        return {"command": cmd, "success": True, "response": "Simulated local answer"}