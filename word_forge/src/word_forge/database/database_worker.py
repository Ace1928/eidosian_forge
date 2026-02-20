"""
Database Worker Module for Word Forge.

This module provides a thread-based worker for continuous database management,
handling operations like maintenance, optimization, backup, and recovery.
It operates asynchronously to ensure database health without blocking the
main application thread.

Architecture:
    ┌────────────────────┐
    │  DatabaseWorker    │
    └──────────┬─────────┘
               │
    ┌──────────┴──────────┐
    │     Components      │
    └─────────────────────┘
    ┌────────┬────────────┐
    │   DB   │  Operation │
    │Manager │  Executor  │
    └────────┴────────────┘
"""

from __future__ import annotations

import datetime
import logging
import random
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, TypedDict, Union, final

from eidosian_core import eidosian

from word_forge.config import config
from word_forge.database.database_manager import DBManager


class DBWorkerState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()
    PAUSED = auto()
    RECOVERY = auto()

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class DBWorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    operation_count: int
    error_count: int
    last_operation: Optional[str]
    last_update: Optional[float]
    uptime: Optional[float]
    state: str
    recent_errors: List[str]
    next_maintenance: Optional[float]
    optimization_level: int
    integrity_status: Optional[str]
    pending_operations: int


class DatabaseError(Exception):
    """Base exception for database worker errors."""

    pass


class MaintenanceError(DatabaseError):
    """Raised when database maintenance operations fail."""

    pass


class OptimizationError(DatabaseError):
    """Raised when database optimization operations fail."""

    pass


class BackupError(DatabaseError):
    """Raised when database backup operations fail."""

    pass


class IntegrityError(DatabaseError):
    """Raised when database integrity check fails."""

    pass


class OperationTimeoutError(DatabaseError):
    """Raised when a database operation exceeds time limit."""

    pass


class DatabaseWorkerInterface(Protocol):
    """Protocol defining the required interface for database workers."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def get_status(self) -> DBWorkerStatus: ...
    def is_alive(self) -> bool: ...
    def run_maintenance(self, wait: bool = False) -> bool: ...
    def run_optimization(self, level: int = 1, wait: bool = False) -> bool: ...
    def run_backup(self, target_path: Optional[str] = None, wait: bool = False) -> bool: ...
    def run_integrity_check(self, wait: bool = False) -> Tuple[bool, Optional[str]]: ...
    def get_metrics(self) -> Dict[str, Any]: ...


@dataclass
class OperationMetrics:
    """
    Tracks metrics for database operations.

    Contains performance data and statistics about database operations
    performed by the worker thread, providing insights for optimization
    and monitoring.

    Attributes:
        operation_count: Total number of operations performed
        error_count: Number of operations that failed
        avg_duration_ms: Average duration of operations in milliseconds
        last_maintenance: Timestamp of last maintenance operation
        last_optimization: Timestamp of last optimization operation
        last_backup: Timestamp of last backup operation
        last_integrity_check: Timestamp of last integrity check
        operation_times: Dictionary mapping operation types to durations
        error_types: Dictionary tracking error occurrences by type
    """

    operation_count: int = 0
    error_count: int = 0
    avg_duration_ms: float = 0.0
    last_maintenance: Optional[float] = None
    last_optimization: Optional[float] = None
    last_backup: Optional[float] = None
    last_integrity_check: Optional[float] = None
    operation_times: Dict[str, List[float]] = field(
        default_factory=lambda: {
            "maintenance": [],
            "optimization": [],
            "backup": [],
            "integrity_check": [],
            "vacuum": [],
            "schema_update": [],
        }
    )
    error_types: Dict[str, int] = field(default_factory=dict)

    @eidosian()
    def record_operation(self, operation_type: str, duration_ms: float, success: bool = True) -> None:
        """
        Record metrics for a completed operation.

        Updates operation counts, durations, and calculates averages
        to maintain accurate performance statistics.

        Args:
            operation_type: Type of operation performed
            duration_ms: Duration of operation in milliseconds
            success: Whether the operation completed successfully
        """
        self.operation_count += 1

        # Update operation-specific metrics
        if operation_type in self.operation_times:
            self.operation_times[operation_type].append(duration_ms)

            # Limit the size of recorded times to avoid unbounded growth
            if len(self.operation_times[operation_type]) > 100:
                self.operation_times[operation_type] = self.operation_times[operation_type][-100:]

        # Record timestamp for specific operation types
        if operation_type == "maintenance":
            self.last_maintenance = time.time()
        elif operation_type == "optimization":
            self.last_optimization = time.time()
        elif operation_type == "backup":
            self.last_backup = time.time()
        elif operation_type == "integrity_check":
            self.last_integrity_check = time.time()

        # Update average duration including all operation types
        total_ops = sum(len(times) for times in self.operation_times.values())
        if total_ops > 0:
            total_duration = sum(sum(times) for times in self.operation_times.values())
            self.avg_duration_ms = total_duration / total_ops

        # Record error if operation failed
        if not success:
            self.error_count += 1

    @eidosian()
    def record_error(self, error_type: str) -> None:
        """
        Record occurrence of an error type.

        Tracks error frequencies to identify problematic patterns
        and prioritize fixes for common issues.

        Args:
            error_type: Type of error that occurred
        """
        self.error_count += 1
        if error_type in self.error_types:
            self.error_types[error_type] += 1
        else:
            self.error_types[error_type] = 1

    @eidosian()
    def get_operation_avg(self, operation_type: str) -> Optional[float]:
        """
        Get average duration for a specific operation type.

        Args:
            operation_type: Type of operation to calculate average for

        Returns:
            Average duration in milliseconds or None if no data
        """
        if operation_type in self.operation_times and self.operation_times[operation_type]:
            return sum(self.operation_times[operation_type]) / len(self.operation_times[operation_type])
        return None

    @eidosian()
    def get_most_common_error(self) -> Optional[Tuple[str, int]]:
        """
        Get the most frequently occurring error.

        Returns:
            Tuple of (error_type, count) or None if no errors
        """
        if not self.error_types:
            return None
        return max(self.error_types.items(), key=lambda x: x[1])

    @eidosian()
    def get_recent_errors(self, limit: int = 5) -> List[str]:
        """
        Get the most frequent recent error types.

        Args:
            limit: Maximum number of error types to return

        Returns:
            List of error type names sorted by frequency
        """
        sorted_errors = sorted(self.error_types.items(), key=lambda x: x[1], reverse=True)
        return [error_type for error_type, _ in sorted_errors[:limit]]


@final
class DatabaseWorker(threading.Thread):
    """
    Manages database maintenance, optimization, and integrity tasks asynchronously.

    This worker runs as a daemon thread that handles essential database operations
    on a regular schedule without blocking the main application. It performs tasks
    such as:

    1. Database maintenance (ANALYZE, reindex)
    2. Performance optimization (VACUUM, optimize indexes)
    3. Database backup and recovery
    4. Integrity verification
    5. Schema migrations

    The worker implements comprehensive lifecycle management including start, stop,
    pause, resume, and restart operations, along with detailed metrics and status
    reporting.

    Attributes:
        db_manager: Database manager providing connection to storage
        poll_interval: Time in seconds between maintenance cycles
        backup_interval: Time in seconds between automatic backups
        backup_path: Directory to store database backups
        metrics: Detailed operational metrics tracking
        logger: Logger for operation and error reporting
    """

    def __init__(
        self,
        db_manager: DBManager,
        poll_interval: Optional[float] = None,
        backup_interval: Optional[float] = None,
        backup_path: Optional[Union[str, Path]] = None,
        optimization_interval: Optional[float] = None,
        daemon: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the database worker.

        Args:
            db_manager: Database manager providing connection access
            poll_interval: Time in seconds between maintenance cycles (defaults to config)
            backup_interval: Time in seconds between automatic backups (defaults to config)
            backup_path: Directory to store database backups (defaults to config)
            optimization_interval: Time in seconds between optimization runs (defaults to config)
            daemon: Whether thread should run as daemon
            logger: Optional logger for detailed logging
        """
        super().__init__(daemon=daemon)
        self.db_manager = db_manager

        # Configuration parameters with defaults from config
        self.poll_interval = poll_interval or getattr(config.database, "poll_interval", 3600.0)
        self.backup_interval = backup_interval or getattr(config.database, "backup_interval", 86400.0)
        self.optimization_interval = optimization_interval or getattr(config.database, "optimization_interval", 43200.0)

        # Set up backup path with fallback to config or default
        if backup_path:
            self.backup_path = Path(backup_path)
        else:
            config_path = getattr(config.database, "backup_path", None)
            if config_path:
                self.backup_path = Path(config_path)
            else:
                # Default: Create a backups directory next to the database
                db_dir = Path(self.db_manager.db_path).parent
                self.backup_path = db_dir / "backups"

        # Create backup directory if it doesn't exist
        self.backup_path.mkdir(parents=True, exist_ok=True)

        # Thread control flags
        self._stop_flag = False
        self._pause_flag = False
        self._wake_event = threading.Event()  # For interruptible sleep (stop or new operation)
        self._status_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self._current_state = DBWorkerState.STOPPED
        self._start_time: Optional[float] = None
        self._last_update: Optional[float] = None

        # Schedule trackers
        self._next_maintenance: Optional[float] = None
        self._next_backup: Optional[float] = None
        self._next_optimization: Optional[float] = None
        self._last_operation: Optional[str] = None

        # Error handling
        self._consecutive_errors = 0
        self._backoff_time = 0.0
        self._error_backoff_base = 30.0  # Base backoff time in seconds
        self._recent_errors: List[str] = []

        # Optimization level (0-3, higher is more aggressive)
        self._optimization_level = 1

        # Integrity status from last check
        self._integrity_status: Optional[str] = None

        # Operation queue for externally-requested operations
        self._pending_operations: List[Tuple[str, Dict[str, Any], Optional[threading.Event]]] = []

        # Performance metrics
        self.metrics = OperationMetrics()

        # Configure logging
        self.logger = logger or logging.getLogger(__name__)

    @eidosian()
    def run(self) -> None:
        """
        Main execution loop that periodically performs database maintenance.

        Executes maintenance, optimization, backup, and other database operations
        according to their scheduled intervals. Also processes any on-demand
        operations requested through the worker interface.
        """
        with self._status_lock:
            self._start_time = time.time()
            self._current_state = DBWorkerState.RUNNING
            self._schedule_operations()

        self.logger.info(
            f"Database worker started: poll={self.poll_interval}s, "
            f"backup={self.backup_interval}s, optimization={self.optimization_interval}s"
        )

        while not self._stop_flag:
            # Clear wake event at start of each iteration to avoid race conditions
            self._wake_event.clear()

            # Skip processing if paused
            if self._pause_flag:
                # Use event.wait() for interruptible sleep during pause
                self._wake_event.wait(timeout=1.0)
                continue

            # Process any pending operations first
            if self._process_pending_operations():
                continue

            try:
                # Check if any scheduled operations are due
                current_time = time.time()

                # Run maintenance if scheduled
                if self._next_maintenance and current_time >= self._next_maintenance:
                    self._execute_maintenance()
                    self._next_maintenance = current_time + self.poll_interval

                # Run backup if scheduled
                if self._next_backup and current_time >= self._next_backup:
                    self._execute_backup()
                    self._next_backup = current_time + self.backup_interval

                # Run optimization if scheduled
                if self._next_optimization and current_time >= self._next_optimization:
                    self._execute_optimization()
                    self._next_optimization = current_time + self.optimization_interval

                # Reset consecutive errors on successful loop
                if self._consecutive_errors > 0:
                    self.logger.info("Database worker recovered from error state")
                    with self._status_lock:
                        self._consecutive_errors = 0
                        self._current_state = DBWorkerState.RUNNING

                # Sleep until next scheduled operation
                next_op_time = min(
                    t
                    for t in [
                        self._next_maintenance,
                        self._next_backup,
                        self._next_optimization,
                    ]
                    if t is not None
                )
                sleep_time = max(0.1, min(next_op_time - time.time(), self.poll_interval))
                # Use event.wait() for interruptible sleep
                self._wake_event.wait(timeout=sleep_time)

            except Exception as e:
                self._handle_error(e)

                # Sleep with exponential backoff during error state (interruptible)
                self._wake_event.wait(timeout=self._backoff_time)

        with self._status_lock:
            self._current_state = DBWorkerState.STOPPED

        self.logger.info("Database worker stopped")

    def _schedule_operations(self) -> None:
        """Schedule initial maintenance, backup, and optimization operations."""
        current_time = time.time()
        self._next_maintenance = current_time + self.poll_interval
        self._next_backup = current_time + self.backup_interval
        self._next_optimization = current_time + self.optimization_interval

    def _process_pending_operations(self) -> bool:
        """
        Process any pending operations in the queue.

        Returns:
            True if an operation was processed, False otherwise
        """
        with self._operation_lock:
            if not self._pending_operations:
                return False

            operation, params, completed_event = self._pending_operations.pop(0)

        try:
            if operation == "maintenance":
                self._execute_maintenance()
            elif operation == "optimization":
                level = params.get("level", 1)
                self._execute_optimization(level)
            elif operation == "backup":
                target_path = params.get("target_path")
                self._execute_backup(target_path)
            elif operation == "integrity_check":
                self._execute_integrity_check()
            else:
                self.logger.warning(f"Unknown operation requested: {operation}")

            # Notify waiter that operation is complete
            if completed_event:
                completed_event.set()

            return True

        except Exception as e:
            self.logger.error(f"Error executing {operation}: {e}")
            if completed_event:
                completed_event.set()
            return True

    def _execute_maintenance(self) -> bool:
        """
        Perform database maintenance operations.

        Executes ANALYZE, reindexes tables, and performs other routine
        maintenance to keep the database running optimally.

        Returns:
            True if maintenance succeeded, False otherwise
        """
        self.logger.info("Starting database maintenance")
        start_time = time.time()

        with self._status_lock:
            self._current_state = DBWorkerState.RUNNING
            self._last_operation = "maintenance"

        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()

            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE")

            # Reindex all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                # Skip system tables
                if table.startswith("sqlite_"):
                    continue

                # Reindex table
                cursor.execute(f"REINDEX '{table}'")

            conn.commit()
            conn.close()

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("maintenance", duration_ms)

            self.logger.info(f"Database maintenance completed in {duration_ms:.2f}ms")
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("maintenance", duration_ms, success=False)
            self.metrics.record_error(type(e).__name__)

            error_msg = f"Database maintenance failed: {e}"
            self.logger.error(error_msg)
            raise MaintenanceError(error_msg) from e

    def _execute_optimization(self, level: int = 1) -> bool:
        """
        Perform database optimization operations.

        Executes VACUUM and other optimizations based on the specified level:
        - Level 1: Basic VACUUM
        - Level 2: VACUUM and incremental optimization
        - Level 3: Full VACUUM and aggressive optimization

        Args:
            level: Optimization aggressiveness (1-3)

        Returns:
            True if optimization succeeded, False otherwise
        """
        self.logger.info(f"Starting database optimization (level {level})")
        start_time = time.time()

        with self._status_lock:
            self._current_state = DBWorkerState.RUNNING
            self._last_operation = "optimization"
            self._optimization_level = level

        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()

            # Basic optimization (always performed)
            cursor.execute("PRAGMA optimize")

            # Level 1+: Basic VACUUM
            if level >= 1:
                cursor.execute("VACUUM")

            # Level 2+: Incremental optimization
            if level >= 2:
                cursor.execute("PRAGMA incremental_vacuum")
                cursor.execute("PRAGMA wal_checkpoint(FULL)")

            # Level 3: Aggressive optimization
            if level >= 3:
                # This rewrites the entire database
                cursor.execute("VACUUM")
                cursor.execute("PRAGMA integrity_check")
                cursor.execute("PRAGMA foreign_key_check")

                # Optimize all indices
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indices = [row[0] for row in cursor.fetchall()]

                for index in indices:
                    cursor.execute(f"REINDEX '{index}'")

            conn.commit()
            conn.close()

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("optimization", duration_ms)

            self.logger.info(f"Database optimization (level {level}) completed in {duration_ms:.2f}ms")
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("optimization", duration_ms, success=False)
            self.metrics.record_error(type(e).__name__)

            error_msg = f"Database optimization (level {level}) failed: {e}"
            self.logger.error(error_msg)
            raise OptimizationError(error_msg) from e

    def _execute_backup(self, target_path: Optional[str] = None) -> bool:
        """
        Perform database backup operation.

        Creates a backup of the database file with timestamp in the filename.
        Optionally compresses the backup to save space.

        Args:
            target_path: Optional custom path for the backup file

        Returns:
            True if backup succeeded, False otherwise
        """
        self.logger.info("Starting database backup")
        start_time = time.time()

        with self._status_lock:
            self._current_state = DBWorkerState.RUNNING
            self._last_operation = "backup"

        try:
            # Generate backup filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            db_filename = Path(self.db_manager.db_path).name
            backup_name = f"{db_filename}_{timestamp}.backup"

            if target_path:
                backup_file = Path(target_path)
            else:
                backup_file = self.backup_path / backup_name

            # Create a database connection for backup
            source_conn = sqlite3.connect(self.db_manager.db_path)

            # Create backup file parent directory if it doesn't exist
            backup_file.parent.mkdir(parents=True, exist_ok=True)

            # Perform the backup using SQLite's backup API
            dest_conn = sqlite3.connect(str(backup_file))
            source_conn.backup(dest_conn)

            # Close connections
            dest_conn.close()
            source_conn.close()

            # Clean up old backups (keep last 10)
            self._cleanup_old_backups()

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("backup", duration_ms)

            self.logger.info(f"Database backup completed in {duration_ms:.2f}ms: {backup_file}")
            return True

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("backup", duration_ms, success=False)
            self.metrics.record_error(type(e).__name__)

            error_msg = f"Database backup failed: {e}"
            self.logger.error(error_msg)
            raise BackupError(error_msg) from e

    def _cleanup_old_backups(self, keep_count: int = 10) -> None:
        """
        Remove old backup files, keeping only the most recent ones.

        Args:
            keep_count: Number of most recent backups to keep
        """
        if not self.backup_path.exists():
            return

        # Find all backup files
        backup_files = list(self.backup_path.glob("*.backup"))

        # Sort by modification time (newest first)
        backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove old backups
        for old_file in backup_files[keep_count:]:
            try:
                old_file.unlink()
                self.logger.debug(f"Removed old backup: {old_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old backup {old_file}: {e}")

    def _execute_integrity_check(self) -> Tuple[bool, Optional[str]]:
        """
        Perform database integrity check.

        Runs SQLite's built-in integrity check to verify database consistency.

        Returns:
            Tuple of (success, error_message)
        """
        self.logger.info("Starting database integrity check")
        start_time = time.time()

        with self._status_lock:
            self._current_state = DBWorkerState.RUNNING
            self._last_operation = "integrity_check"

        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            results = cursor.fetchall()

            # Check foreign keys
            cursor.execute("PRAGMA foreign_key_check")
            fk_results = cursor.fetchall()

            conn.close()

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("integrity_check", duration_ms)

            # Check results
            if len(results) == 1 and results[0][0] == "ok" and not fk_results:
                integrity_status = "ok"
                self.logger.info(f"Database integrity check passed in {duration_ms:.2f}ms")
            else:
                # Combine all integrity errors
                integrity_errors = [row[0] for row in results if row[0] != "ok"]
                fk_errors = [f"Foreign key violation in table {row[0]}" for row in fk_results]
                all_errors = integrity_errors + fk_errors

                if all_errors:
                    integrity_status = f"Errors: {', '.join(all_errors)}"
                    self.logger.warning(f"Database integrity check found issues: {integrity_status}")
                else:
                    integrity_status = "unknown"
                    self.logger.warning("Database integrity check returned unexpected results")

            with self._status_lock:
                self._integrity_status = integrity_status

            return integrity_status == "ok", integrity_status

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_operation("integrity_check", duration_ms, success=False)
            self.metrics.record_error(type(e).__name__)

            error_msg = f"Database integrity check failed: {e}"
            self.logger.error(error_msg)

            with self._status_lock:
                self._integrity_status = f"Check failed: {e}"

            raise IntegrityError(error_msg) from e

    def _handle_error(self, error: Exception) -> None:
        """
        Handle worker loop errors with exponential backoff.

        Args:
            error: The exception that occurred
        """
        with self._status_lock:
            self._consecutive_errors += 1
            self._current_state = DBWorkerState.ERROR

            # Calculate exponential backoff with jitter
            backoff_base = self._error_backoff_base * (2 ** min(6, self._consecutive_errors - 1))
            jitter = random.uniform(0, backoff_base * 0.1)
            self._backoff_time = min(300, backoff_base + jitter)  # Cap at 5 minutes

            # Record error details
            error_type = type(error).__name__
            self.metrics.record_error(error_type)
            self._recent_errors = self.metrics.get_recent_errors()

        # Log error message
        self.logger.error(f"Database worker error ({error_type}): {error}. " f"Retrying in {self._backoff_time:.1f}s")

        # Log detailed traceback at debug level
        self.logger.debug(f"Error traceback: {traceback.format_exc()}")

    @eidosian()
    def stop(self) -> None:
        """Signal the worker to stop and wait for any current operation to complete."""
        self.logger.info("Database worker stopping...")

        with self._status_lock:
            self._stop_flag = True
            self._current_state = DBWorkerState.STOPPED

            # Cancel any pending operations
            with self._operation_lock:
                self._pending_operations.clear()

            # Add a more aggressive interrupt flag
            self._immediate_stop = True

        # Signal the wake event to wake up any sleeping thread
        self._wake_event.set()

    @eidosian()
    def restart(self) -> None:
        """
        Restart the worker thread.

        Stops the current thread if running, then starts a new thread.
        This is useful for applying configuration changes or recovering
        from error states.
        """
        self.logger.info("Database worker restarting...")

        # Stop current thread if running
        if self.is_alive():
            self._stop_flag = True
            self.join(timeout=30.0)  # Wait up to 30 seconds for clean stop

        # Reset control flags
        self._stop_flag = False
        self._pause_flag = False

        # Start new thread
        self.start()

    @eidosian()
    def pause(self) -> None:
        """
        Pause worker execution without stopping the thread.

        Sets the pause flag to temporarily suspend operations while
        keeping the thread alive. Useful for administrative tasks
        that need exclusive database access.
        """
        with self._status_lock:
            self._pause_flag = True
            self._current_state = DBWorkerState.PAUSED

        self.logger.info("Database worker paused")

    @eidosian()
    def resume(self) -> None:
        """
        Resume worker execution after being paused.

        Clears the pause flag to continue normal operations
        from the paused state.
        """
        with self._status_lock:
            self._pause_flag = False
            self._current_state = DBWorkerState.RUNNING

        self.logger.info("Database worker resumed")

    @eidosian()
    def get_status(self) -> DBWorkerStatus:
        """
        Return the current status of the database worker.

        Provides comprehensive status information including operational
        metrics, error counts, and scheduled maintenance times.

        Returns:
            Dictionary containing detailed worker status
        """
        with self._status_lock:
            uptime = None
            if self._start_time:
                uptime = time.time() - self._start_time

            return {
                "running": self.is_alive() and not self._stop_flag and not self._pause_flag,
                "operation_count": self.metrics.operation_count,
                "error_count": self.metrics.error_count,
                "last_operation": self._last_operation,
                "last_update": self._last_update,
                "next_maintenance": self._next_maintenance,
                "uptime": uptime,
                "state": str(self._current_state),
                "recent_errors": self._recent_errors,
                "optimization_level": self._optimization_level,
                "integrity_status": self._integrity_status,
                "pending_operations": len(self._pending_operations),
            }

    @eidosian()
    def run_maintenance(self, wait: bool = False) -> bool:
        """
        Schedule an immediate maintenance operation.

        Queues a maintenance operation to be executed as soon as possible,
        optionally waiting for completion.

        Args:
            wait: If True, block until operation completes

        Returns:
            True if operation was scheduled successfully
        """
        self.logger.info("Scheduling immediate maintenance operation")

        completion_event = threading.Event() if wait else None

        with self._operation_lock:
            self._pending_operations.append(("maintenance", {}, completion_event))

        # Wake up the worker thread to process the operation immediately
        self._wake_event.set()

        if wait and completion_event:
            self.logger.debug("Waiting for maintenance operation to complete")
            completion_event.wait(timeout=300)  # Wait up to 5 minutes
            self.logger.debug("Maintenance operation completed")

        return True

    @eidosian()
    def run_optimization(self, level: int = 1, wait: bool = False) -> bool:
        """
        Schedule an immediate optimization operation.

        Queues an optimization operation to be executed as soon as possible,
        optionally waiting for completion.

        Args:
            level: Optimization level (1-3, higher is more aggressive)
            wait: If True, block until operation completes

        Returns:
            True if operation was scheduled successfully
        """
        # Validate optimization level
        if level < 1 or level > 3:
            level = 1

        self.logger.info(f"Scheduling immediate optimization operation (level {level})")

        completion_event = threading.Event() if wait else None

        with self._operation_lock:
            self._pending_operations.append(("optimization", {"level": level}, completion_event))

        # Wake up the worker thread to process the operation immediately
        self._wake_event.set()

        if wait and completion_event:
            self.logger.debug("Waiting for optimization operation to complete")
            completion_event.wait(timeout=600)  # Wait up to 10 minutes
            self.logger.debug("Optimization operation completed")

        return True

    @eidosian()
    def run_backup(self, target_path: Optional[str] = None, wait: bool = False) -> bool:
        """
        Schedule an immediate backup operation.

        Queues a backup operation to be executed as soon as possible,
        optionally waiting for completion.

        Args:
            target_path: Optional custom path for the backup file
            wait: If True, block until operation completes

        Returns:
            True if operation was scheduled successfully
        """
        self.logger.info("Scheduling immediate backup operation")

        completion_event = threading.Event() if wait else None

        with self._operation_lock:
            self._pending_operations.append(("backup", {"target_path": target_path}, completion_event))

        # Wake up the worker thread to process the operation immediately
        self._wake_event.set()

        if wait and completion_event:
            self.logger.debug("Waiting for backup operation to complete")
            completion_event.wait(timeout=300)  # Wait up to 5 minutes
            self.logger.debug("Backup operation completed")

        return True

    @eidosian()
    def run_integrity_check(self, wait: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Schedule an immediate integrity check operation.

        Queues an integrity check operation to be executed as soon as possible,
        optionally waiting for completion.

        Args:
            wait: If True, block until operation completes

        Returns:
            Tuple of (success, integrity_status)
        """
        self.logger.info("Scheduling immediate integrity check operation")

        completion_event = threading.Event() if wait else None

        with self._operation_lock:
            self._pending_operations.append(("integrity_check", {}, completion_event))

        # Wake up the worker thread to process the operation immediately
        self._wake_event.set()

        if wait and completion_event:
            self.logger.debug("Waiting for integrity check operation to complete")
            completion_event.wait(timeout=300)  # Wait up to 5 minutes
            self.logger.debug("Integrity check operation completed")

            with self._status_lock:
                integrity_ok = self._integrity_status == "ok"
                return integrity_ok, self._integrity_status

        return True, None

    @eidosian()
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed worker performance metrics.

        Returns:
            Dictionary containing comprehensive metrics and statistics
        """
        metrics_dict: Dict[str, Any] = {
            "operation_count": self.metrics.operation_count,
            "error_count": self.metrics.error_count,
            "avg_duration_ms": self.metrics.avg_duration_ms,
            "maintenance": {
                "last_run": self.metrics.last_maintenance,
                "avg_duration_ms": self.metrics.get_operation_avg("maintenance"),
                "count": len(self.metrics.operation_times.get("maintenance", [])),
            },
            "optimization": {
                "last_run": self.metrics.last_optimization,
                "avg_duration_ms": self.metrics.get_operation_avg("optimization"),
                "count": len(self.metrics.operation_times.get("optimization", [])),
                "current_level": self._optimization_level,
            },
            "backup": {
                "last_run": self.metrics.last_backup,
                "avg_duration_ms": self.metrics.get_operation_avg("backup"),
                "count": len(self.metrics.operation_times.get("backup", [])),
                "backup_path": str(self.backup_path),
            },
            "integrity": {
                "last_run": self.metrics.last_integrity_check,
                "avg_duration_ms": self.metrics.get_operation_avg("integrity_check"),
                "count": len(self.metrics.operation_times.get("integrity_check", [])),
                "status": self._integrity_status,
            },
            "errors": {
                "count": self.metrics.error_count,
                "types": dict(self.metrics.error_types),
                "most_common": self.metrics.get_most_common_error(),
            },
            "schedule": {
                "next_maintenance": self._next_maintenance,
                "next_backup": self._next_backup,
                "next_optimization": self._next_optimization,
            },
            "uptime": None,
        }

        # Add uptime if available
        if self._start_time:
            metrics_dict["uptime"] = time.time() - self._start_time

        return metrics_dict
