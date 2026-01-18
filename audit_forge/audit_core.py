from pathlib import Path
import time
from .coverage import CoverageTracker
from .tasks import IdempotentTaskManager


class AuditForge:
    """
    The central orchestration point for system audits and review cycles.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.coverage = CoverageTracker(data_dir / "coverage_map.json")
        self.todo_manager = IdempotentTaskManager(Path("/home/lloyd/TODO.md"))
        self.roadmap_manager = IdempotentTaskManager(
            Path("/home/lloyd/eidosian_forge/eidosian_roadmap.md")
        )

    def start_review_session(self, agent_id: str):
        self.current_session = {"agent": agent_id, "start": time.time()}

    def verify_coverage(self, target_root: str) -> dict:
        unreviewed = self.coverage.get_unreviewed_files(Path(target_root))
        return {
            "root": target_root,
            "unreviewed_count": len(unreviewed),
            "unreviewed_sample": unreviewed[:10],
        }
