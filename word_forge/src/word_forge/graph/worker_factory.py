from __future__ import annotations

from typing import TYPE_CHECKING

from eidosian_core import eidosian

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from .graph_worker import GraphWorker


@eidosian()
def restart_worker(worker: "GraphWorker") -> "GraphWorker":
    """Stop ``worker`` and return a new running ``GraphWorker`` instance."""
    worker.logger.info("Restarting GraphWorker via factory helper...")
    worker.stop()
    worker.join(timeout=worker.poll_interval + 5.0)
    if worker.is_alive():
        worker.logger.warning("Worker thread did not terminate cleanly during restart.")

    from .graph_worker import GraphWorker  # Local import to avoid circular dependency

    new_worker = GraphWorker(
        graph_manager=worker.graph_manager,
        poll_interval=worker.poll_interval,
        output_path=worker.output_path,
        visualization_path=worker.visualization_path,
        daemon=worker.daemon,
    )
    new_worker.start()
    return new_worker
