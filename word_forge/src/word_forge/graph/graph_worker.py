import logging
import random
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, TypedDict, final

from eidosian_core import eidosian

from word_forge.config import config
from word_forge.exceptions import (  # Import specific exceptions
    GraphDataError,
    GraphError,
    GraphIOError,
    GraphLayoutError,
    GraphVisualizationError,
)
from word_forge.graph.graph_builder import GraphUpdateMetrics
from word_forge.graph.graph_manager import GraphManager


class WorkerState(Enum):
    """Defined states for the graph worker."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()
    PAUSED = auto()

    def __str__(self) -> str:
        """Return the lowercase name of the state."""
        return self.name.lower()


class WorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    update_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str
    last_error: Optional[str]
    last_new_nodes: int
    last_new_edges: int
    last_processed_words: int
    last_full_rebuild: bool


# Define specific GraphWorker exceptions inheriting from GraphError if needed,
# or use the ones from word_forge.exceptions
class GraphSaveError(GraphIOError):  # Inherit from GraphIOError
    """Raised when a graph save operation fails."""

    pass


class GraphUpdateError(GraphError):  # Keep inheriting from GraphError
    """Raised when a graph update operation fails."""

    pass


# GraphVisualizationError is already defined in exceptions


class GraphDirectoryError(GraphIOError):  # Inherit from GraphIOError
    """Raised when a directory operation fails."""

    pass


class GraphWorkerInterface(Protocol):
    """Protocol defining the required interface for a graph worker."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_status(self) -> WorkerStatus: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def is_alive(self) -> bool: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def restart(self) -> None: ...
    def run(self) -> None: ...


@final
class GraphWorker(threading.Thread):
    """
    Background worker that maintains the lexical graph representation.

    Periodically rebuilds or updates the in-memory graph and saves it to disk
    for visualization and further queries. Thread-safe operations ensure graph
    integrity across concurrent access patterns.

    Attributes:
        graph_manager: Manager handling graph operations.
        poll_interval: Seconds between graph update cycles.
        output_path: Path where graph will be saved (GEXF format).
        visualization_path: Path where visualization HTML will be saved.
        logger: Logger instance for the worker.
    """

    def __init__(
        self,
        graph_manager: GraphManager,
        poll_interval: Optional[float] = None,
        output_path: Optional[str] = None,
        visualization_path: Optional[str] = None,
        daemon: bool = True,
    ) -> None:
        """
        Initialize the graph worker with configuration parameters.

        Args:
            graph_manager: Graph manager instance to perform operations.
            poll_interval: Seconds between update cycles (defaults to config).
            output_path: Path where the graph will be saved (defaults to config).
            visualization_path: Path where visualization will be saved (defaults to config).
            daemon: Whether thread should be daemonic (auto-terminate when main exits).
        """
        super().__init__(daemon=daemon, name="GraphWorkerThread")  # Give the thread a name
        self.graph_manager = graph_manager
        self.logger = logging.getLogger(__name__)

        # Use config values if parameters not provided, with fallbacks
        cfg = config.graph  # Shortcut to graph config
        self.poll_interval = (
            poll_interval
            if poll_interval is not None
            else (cfg.animation_duration_ms / 1000.0 if cfg.animation_duration_ms > 0 else 30.0)
        )

        # Ensure proper paths for output and visualization
        # Default GEXF export path
        default_gexf_path = cfg.get_export_path / "lexical_graph.gexf"
        self.output_path = output_path or str(default_gexf_path)

        # Default visualization HTML path
        vis_path_str = visualization_path or cfg.visualization_path
        vis_path = Path(vis_path_str)
        if vis_path.suffix.lower() != ".html":  # Ensure HTML extension
            vis_path = vis_path / "lexical_graph.html" if vis_path.is_dir() else vis_path.with_suffix(".html")
        self.visualization_path = str(vis_path)

        # Internal state and control flags
        self._stop_event = threading.Event()  # Use Event for clearer stop signal
        self._pause_event = threading.Event()  # Use Event for pause/resume
        self._last_update: Optional[float] = None
        self._start_time: Optional[float] = None
        self._update_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._current_state = WorkerState.STOPPED
        self._status_lock = threading.RLock()  # Reentrant lock for status access
        self._error_backoff = self.poll_interval  # Initial backoff delay
        self._last_cycle_metrics: GraphUpdateMetrics = GraphUpdateMetrics()

        self.logger.debug(
            f"GraphWorker initialized: Poll={self.poll_interval}s, Output='{self.output_path}', Viz='{self.visualization_path}'"
        )

    @eidosian()
    def pause(self) -> None:
        """Pause worker execution without stopping the thread."""
        with self._status_lock:
            if self._current_state == WorkerState.RUNNING:
                self._pause_event.set()  # Signal pause
                self._current_state = WorkerState.PAUSED
                self.logger.info("GraphWorker paused")
            elif self._current_state == WorkerState.PAUSED:
                self.logger.info("GraphWorker already paused")
            else:
                self.logger.warning(f"Cannot pause worker in state: {self._current_state}")

    @eidosian()
    def resume(self) -> None:
        """Resume worker execution after being paused."""
        with self._status_lock:
            if self._current_state == WorkerState.PAUSED:
                self._pause_event.clear()  # Clear pause signal
                self._current_state = WorkerState.RUNNING
                self.logger.info("GraphWorker resumed")
            elif self._current_state == WorkerState.RUNNING:
                self.logger.info("GraphWorker already running")
            else:
                self.logger.warning(f"Cannot resume worker in state: {self._current_state}")

    @eidosian()
    def restart(self) -> "GraphWorker":
        """Restart the worker and return the new running instance."""
        from .worker_factory import restart_worker

        return restart_worker(self)

    @eidosian()
    def run(self) -> None:
        """
        Main execution loop that periodically updates and saves the graph.

        Updates occur at intervals defined by poll_interval. Handles stop/pause
        signals and recovers from errors with backoff.
        """
        with self._status_lock:
            if self._current_state != WorkerState.STOPPED:
                self.logger.warning("Worker already running or in an unexpected state.")
                return  # Avoid starting multiple times conceptually
            self._start_time = time.time()
            self._current_state = WorkerState.RUNNING
            self._stop_event.clear()  # Ensure stop is not set initially
            self._pause_event.clear()  # Ensure pause is not set initially

        self.logger.info(f"GraphWorker thread '{self.name}' started.")

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                # Wait efficiently while paused
                self._stop_event.wait(1.0)  # Check stop event periodically even when paused
                continue

            cycle_start_time = time.time()
            try:
                self._execute_update_cycle()
                # Successful cycle, reset backoff
                with self._status_lock:
                    self._error_backoff = self.poll_interval
                    self._last_error = None  # Clear last error on success
                    if not self._pause_event.is_set():  # Don't overwrite PAUSED state
                        self._current_state = WorkerState.RUNNING

            except Exception as e:
                self._handle_execution_error(e)
                # Apply backoff after error
                wait_time = self._error_backoff
                self._trigger_backoff()  # Increase backoff for next potential error
                self._stop_event.wait(wait_time)  # Wait using backoff duration
                continue  # Skip normal sleep, already waited

            # Calculate remaining time and wait, checking stop event
            elapsed = time.time() - cycle_start_time
            wait_duration = max(0, self.poll_interval - elapsed)
            self._stop_event.wait(wait_duration)  # Wait for the remaining interval or until stopped

        # --- Thread Termination ---
        with self._status_lock:
            self._current_state = WorkerState.STOPPED
            uptime = time.time() - self._start_time if self._start_time else 0

        self.logger.info(
            f"GraphWorker thread '{self.name}' stopped after {uptime:.2f}s. "
            f"Updates: {self._update_count}, Errors: {self._error_count}."
        )

    def _execute_update_cycle(self) -> None:
        """Execute a complete update cycle: prepare, update, save, visualize."""
        self.logger.debug(f"Starting update cycle #{self._update_count + 1}")
        start_ts = time.time()

        # 1. Ensure directories exist
        try:
            self._ensure_output_directories()
        except Exception as e:
            # Wrap directory errors
            raise GraphDirectoryError(f"Failed to ensure output directories exist: {e}", e) from e

        # 2. Verify DB tables (optional, can be intensive)
        # Consider doing this less frequently if performance is an issue
        # if not self._verify_database_tables():
        #     self.logger.warning("DB verification failed. Skipping update cycle.")
        #     raise GraphDataError("Required database tables missing or inaccessible.", None)

        # 3. Update Graph (using build_graph for simplicity now)
        try:
            self._update_graph()
        except (GraphDataError, GraphError) as e:
            raise GraphUpdateError(f"Graph update failed: {e}", e) from e
        except Exception as e:  # Catch unexpected errors during update
            raise GraphUpdateError(f"Unexpected error during graph update: {e}", e) from e

        # 4. Save Graph
        try:
            self._save_graph()
        except (GraphIOError, GraphError) as e:
            raise GraphSaveError(f"Graph save failed: {e}", e) from e
        except Exception as e:  # Catch unexpected errors during save
            raise GraphSaveError(f"Unexpected error during graph save: {e}", e) from e

        # 5. Generate Visualization (non-critical, log warnings)
        try:
            self._generate_visualization()
        except (GraphVisualizationError, GraphLayoutError, ImportError) as e:
            self.logger.warning(f"Visualization generation failed: {e}")
            # Optionally store this as a non-fatal error in status
        except Exception as e:  # Catch unexpected errors during visualization
            self.logger.warning(
                f"Unexpected error during visualization: {e}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG),
            )

        # Track successful update
        with self._status_lock:
            self._last_update = time.time()
            self._update_count += 1

        self.logger.debug(f"Update cycle #{self._update_count} completed in {time.time() - start_ts:.3f}s.")

    def _verify_database_tables(self) -> bool:
        """Verify that required database tables exist."""
        try:
            return self.graph_manager.verify_database_tables()
        except Exception as e:
            self.logger.error(
                f"Database verification failed: {e}",
                exc_info=self.logger.isEnabledFor(logging.DEBUG),
            )
            return False

    def _handle_graph_error(self, error: Exception) -> None:
        """Handles errors specifically during the graph update phase."""
        # This method seems redundant if _handle_execution_error covers it.
        # Kept for potential specific logic later.
        self.logger.error(
            f"Graph operation error: {error}",
            exc_info=self.logger.isEnabledFor(logging.DEBUG),
        )
        # No backoff trigger here, handled in the main loop's exception block

    def _trigger_backoff(self) -> None:
        """Increase the error backoff delay exponentially."""
        # Increase delay, add jitter, cap at a max value (e.g., 5 minutes)
        max_backoff = 300.0
        self._error_backoff = min(max_backoff, self._error_backoff * 1.5 + random.uniform(0, 2))
        self.logger.warning(f"Increasing error backoff to {self._error_backoff:.1f}s")

    def _handle_execution_error(self, error: Exception) -> None:
        """Log execution errors and update worker state."""
        error_type = type(error).__name__
        error_message = str(error)
        detailed_error = f"{error_type}: {error_message}"

        with self._status_lock:
            self._error_count += 1
            self._current_state = WorkerState.ERROR
            self._last_error = detailed_error

        self.logger.error(
            f"Error during update cycle: {detailed_error}",
            exc_info=self.logger.isEnabledFor(logging.DEBUG),
        )
        # Backoff is triggered in the main loop after this handler

    def _ensure_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        paths_to_check = [self.output_path, self.visualization_path]
        for file_path_str in paths_to_check:
            dir_path = Path(file_path_str).parent
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"Created directory: {dir_path}")
                except OSError as e:
                    # Raise a specific error if directory creation fails
                    raise GraphDirectoryError(f"Failed to create directory {dir_path}: {e}", e) from e

    def _update_graph(self) -> None:
        """Update the graph data structure using incremental refresh when possible."""

        if self.graph_manager.get_node_count() == 0:
            self.logger.debug("Graph empty; performing full build.")
            self.graph_manager.build_graph()
        else:
            self.logger.debug("Graph initialized; performing incremental update.")
            self.graph_manager.update_graph()

        metrics = self.graph_manager.get_last_update_metrics()
        with self._status_lock:
            self._last_cycle_metrics = metrics

        self.logger.debug("Graph update complete. Metrics: %s", metrics)

    def _save_graph(self) -> None:
        """Save the graph to a GEXF file."""
        if self.graph_manager.get_node_count() > 0:
            self.logger.debug(f"Saving graph to {self.output_path}...")
            self.graph_manager.save_to_gexf(self.output_path)
            self.logger.debug("Graph save complete.")
        else:
            self.logger.debug("Skipping save for empty graph.")

    def _generate_visualization(self) -> None:
        """Generate graph visualization HTML file."""
        if self.graph_manager.get_node_count() > 0:
            self.logger.debug(f"Generating visualization to {self.visualization_path}...")
            # Use the manager's default visualize method (which chooses 2D/3D)
            self.graph_manager.visualize(output_path=self.visualization_path, open_in_browser=False)
            self.logger.debug("Visualization generation complete.")
        else:
            self.logger.debug("Skipping visualization for empty graph.")

    @eidosian()
    def stop(self) -> None:
        """Signal the worker thread to stop gracefully."""
        if not self._stop_event.is_set():
            self.logger.info(f"Signaling GraphWorker thread '{self.name}' to stop...")
            self._stop_event.set()
            # If paused, clear pause so the loop can see the stop signal
            if self._pause_event.is_set():
                self._pause_event.clear()
        else:
            self.logger.debug(f"GraphWorker thread '{self.name}' already stopping.")

    @eidosian()
    def get_status(self) -> WorkerStatus:
        """Return the current operational status of the worker."""
        with self._status_lock:
            current_time = time.time()
            uptime = (current_time - self._start_time) if self._start_time else None
            is_running = self.is_alive() and not self._stop_event.is_set()

            # Refine state reporting based on events
            state = self._current_state
            if is_running and self._pause_event.is_set():
                state = WorkerState.PAUSED
            elif is_running and state != WorkerState.ERROR:  # Don't overwrite ERROR state if running
                state = WorkerState.RUNNING
            elif not is_running:
                state = WorkerState.STOPPED

            status: WorkerStatus = {
                "running": is_running and state == WorkerState.RUNNING,  # True only if actively running
                "update_count": self._update_count,
                "error_count": self._error_count,
                "last_update": self._last_update,
                "uptime": uptime,
                "state": str(state),
                "last_error": self._last_error,
                "last_new_nodes": self._last_cycle_metrics.new_nodes,
                "last_new_edges": self._last_cycle_metrics.new_edges,
                "last_processed_words": self._last_cycle_metrics.processed_words,
                "last_full_rebuild": self._last_cycle_metrics.full_rebuild,
            }
            return status

    @eidosian()
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and state metrics for the graph worker."""
        with self._status_lock:
            # Return a copy to prevent external modification
            return {
                "update_count": self._update_count,
                "error_count": self._error_count,
                "last_update_timestamp": self._last_update,
                "last_error_message": self._last_error,
                "current_backoff_s": (self._error_backoff if self._current_state == WorkerState.ERROR else 0),
                "last_new_nodes": self._last_cycle_metrics.new_nodes,
                "last_new_edges": self._last_cycle_metrics.new_edges,
                "last_processed_words": self._last_cycle_metrics.processed_words,
                "last_full_rebuild": self._last_cycle_metrics.full_rebuild,
                "last_refresh_watermark": self._last_cycle_metrics.max_last_refreshed,
            }
