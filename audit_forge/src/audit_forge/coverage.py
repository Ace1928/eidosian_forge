import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from eidosian_core import eidosian


class CoverageTracker:
    """
    Tracks audit and review coverage across the filesystem.
    Ensures that every file and folder is accounted for in the Eidosian ecosystem.
    """

    def __init__(self, persistence_path: Path):
        self.persistence_path = persistence_path
        self.data: Dict[str, Any] = {"coverage": {}, "last_full_audit": None}
        self.load()

    @eidosian()
    def load(self):
        if self.persistence_path.exists():
            try:
                self.data = json.loads(self.persistence_path.read_text())
            except json.JSONDecodeError:
                pass

    @eidosian()
    def save(self):
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self.persistence_path.write_text(json.dumps(self.data, indent=2))

    @eidosian()
    def mark_reviewed(self, path: str, agent_id: str, scope: str = "shallow"):
        """
        Mark a path as reviewed.
        Scope can be 'shallow', 'deep', or 'exhaustive'.
        """
        self.data["coverage"][path] = {
            "last_reviewed": time.time(),
            "agent": agent_id,
            "scope": scope,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save()

    @eidosian()
    def get_coverage_status(self, path: str) -> Optional[Dict[str, Any]]:
        return self.data["coverage"].get(path)

    @eidosian()
    def get_unreviewed_files(
        self, root_path: Path, ignore_patterns: List[str] = None
    ) -> List[str]:
        """
        Compare the filesystem against the coverage map.
        """
        unreviewed = []
        # Ensure root_path is absolute for consistency
        root_path = root_path.resolve()
        for p in root_path.rglob("*"):
            if p.is_file():
                # We store paths relative to the system root or a stable base in production,
                # but for this tool, let's use the provided root_path as the anchor.
                rel_path = str(p.relative_to(root_path))
                if rel_path not in self.data["coverage"]:
                    unreviewed.append(rel_path)
        return unreviewed
