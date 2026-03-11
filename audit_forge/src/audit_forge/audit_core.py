"""
Eidosian Audit Forge Core.
Orchestrates systemic self-auditing, coverage tracking, and task management.
"""
from pathlib import Path
import time
from .coverage import CoverageTracker
from .tasks import IdempotentTaskManager
from eidosian_core import eidosian

try:
    import global_info
    ROOT_DIR = global_info.ROOT_DIR
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parent.parent

class AuditForge:
    """
    The central orchestration point for Eidosian system audits and review cycles.
    
    Integrates coverage tracking with idempotent markdown task management to ensure 
    no review operation is lost or redundantly applied.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize the Audit Forge with its persistence substrate.
        
        Args:
            data_dir (Path): Directory where audit state (e.g., coverage maps) is stored.
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.coverage = CoverageTracker(data_dir / "coverage_map.json")
        self.todo_manager = IdempotentTaskManager(ROOT_DIR / "TODO.md")
        self.roadmap_manager = IdempotentTaskManager(
            ROOT_DIR / "eidosian_roadmap.md"
        )

    @eidosian()
    def start_review_session(self, agent_id: str):
        """
        Instantiates a new review session context.
        
        Args:
            agent_id (str): Unique identifier for the auditing entity.
        """
        self.current_session = {"agent": agent_id, "start": time.time()}

    @eidosian()
    def verify_coverage(self, target_root: str) -> dict:
        """
        Scans the target root and reports files that lack a verified review stamp.
        
        Args:
            target_root (str): The filesystem path to audit.
            
        Returns:
            dict: Summary metrics containing unreviewed count and sample paths.
        """
        unreviewed = self.coverage.get_unreviewed_files(Path(target_root))
        return {
            "root": target_root,
            "unreviewed_count": len(unreviewed),
            "unreviewed_sample": unreviewed[:10],
        }
