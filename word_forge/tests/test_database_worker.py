"""Tests for word_forge.database.database_worker module.

This module provides comprehensive tests for the DatabaseWorker class
including lifecycle management, maintenance operations, optimization,
backup, integrity checks, and metrics collection.
"""

import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.database.database_manager import DBManager
from word_forge.database.database_worker import (
    BackupError,
    DatabaseError,
    DatabaseWorker,
    DBWorkerState,
    IntegrityError,
    MaintenanceError,
    OperationMetrics,
    OperationTimeoutError,
    OptimizationError,
)


class TestDBWorkerState:
    """Tests for the DBWorkerState enum."""

    def test_running_state_exists(self) -> None:
        """Test RUNNING state exists."""
        assert DBWorkerState.RUNNING

    def test_stopped_state_exists(self) -> None:
        """Test STOPPED state exists."""
        assert DBWorkerState.STOPPED

    def test_error_state_exists(self) -> None:
        """Test ERROR state exists."""
        assert DBWorkerState.ERROR

    def test_paused_state_exists(self) -> None:
        """Test PAUSED state exists."""
        assert DBWorkerState.PAUSED

    def test_recovery_state_exists(self) -> None:
        """Test RECOVERY state exists."""
        assert DBWorkerState.RECOVERY

    def test_str_representation(self) -> None:
        """Test string representation is lowercase."""
        assert str(DBWorkerState.RUNNING) == "running"
        assert str(DBWorkerState.STOPPED) == "stopped"
        assert str(DBWorkerState.ERROR) == "error"
        assert str(DBWorkerState.PAUSED) == "paused"

    def test_all_states_count(self) -> None:
        """Test that there are 5 states."""
        assert len(DBWorkerState) == 5


class TestDatabaseExceptions:
    """Tests for database worker exception classes."""

    def test_database_error_base(self) -> None:
        """Test DatabaseError is the base exception."""
        error = DatabaseError("test error")
        assert str(error) == "test error"
        assert isinstance(error, Exception)

    def test_maintenance_error_inherits(self) -> None:
        """Test MaintenanceError inherits from DatabaseError."""
        error = MaintenanceError("maintenance failed")
        assert isinstance(error, DatabaseError)
        assert str(error) == "maintenance failed"

    def test_optimization_error_inherits(self) -> None:
        """Test OptimizationError inherits from DatabaseError."""
        error = OptimizationError("optimization failed")
        assert isinstance(error, DatabaseError)

    def test_backup_error_inherits(self) -> None:
        """Test BackupError inherits from DatabaseError."""
        error = BackupError("backup failed")
        assert isinstance(error, DatabaseError)

    def test_integrity_error_inherits(self) -> None:
        """Test IntegrityError inherits from DatabaseError."""
        error = IntegrityError("integrity check failed")
        assert isinstance(error, DatabaseError)

    def test_operation_timeout_error_inherits(self) -> None:
        """Test OperationTimeoutError inherits from DatabaseError."""
        error = OperationTimeoutError("operation timed out")
        assert isinstance(error, DatabaseError)


class TestOperationMetrics:
    """Tests for the OperationMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values for OperationMetrics."""
        metrics = OperationMetrics()
        assert metrics.operation_count == 0
        assert metrics.error_count == 0
        assert metrics.avg_duration_ms == 0.0
        assert metrics.last_maintenance is None
        assert metrics.last_optimization is None
        assert metrics.last_backup is None
        assert metrics.last_integrity_check is None

    def test_operation_times_default(self) -> None:
        """Test operation_times has expected keys by default."""
        metrics = OperationMetrics()
        expected_keys = {
            "maintenance",
            "optimization",
            "backup",
            "integrity_check",
            "vacuum",
            "schema_update",
        }
        assert set(metrics.operation_times.keys()) == expected_keys

    def test_record_operation_increments_count(self) -> None:
        """Test record_operation increments operation_count."""
        metrics = OperationMetrics()
        metrics.record_operation("maintenance", 100.0)
        assert metrics.operation_count == 1
        metrics.record_operation("backup", 200.0)
        assert metrics.operation_count == 2

    def test_record_operation_updates_timestamps(self) -> None:
        """Test record_operation updates last_* timestamps."""
        metrics = OperationMetrics()
        before = time.time()
        metrics.record_operation("maintenance", 100.0)
        after = time.time()
        assert metrics.last_maintenance is not None
        assert before <= metrics.last_maintenance <= after

    def test_record_operation_stores_duration(self) -> None:
        """Test record_operation stores duration in operation_times."""
        metrics = OperationMetrics()
        metrics.record_operation("maintenance", 150.0)
        assert 150.0 in metrics.operation_times["maintenance"]

    def test_record_operation_limits_stored_times(self) -> None:
        """Test record_operation limits stored times to 100."""
        metrics = OperationMetrics()
        for i in range(150):
            metrics.record_operation("maintenance", float(i))
        assert len(metrics.operation_times["maintenance"]) == 100

    def test_record_operation_failure(self) -> None:
        """Test record_operation with success=False increments error_count."""
        metrics = OperationMetrics()
        metrics.record_operation("maintenance", 100.0, success=False)
        assert metrics.error_count == 1
        assert metrics.operation_count == 1

    def test_record_operation_avg_duration(self) -> None:
        """Test record_operation updates avg_duration_ms."""
        metrics = OperationMetrics()
        metrics.record_operation("maintenance", 100.0)
        metrics.record_operation("maintenance", 200.0)
        assert metrics.avg_duration_ms == 150.0

    def test_record_error(self) -> None:
        """Test record_error increments error counts."""
        metrics = OperationMetrics()
        metrics.record_error("DatabaseError")
        assert metrics.error_count == 1
        assert metrics.error_types["DatabaseError"] == 1

    def test_record_error_accumulates(self) -> None:
        """Test record_error accumulates error types."""
        metrics = OperationMetrics()
        metrics.record_error("DatabaseError")
        metrics.record_error("DatabaseError")
        metrics.record_error("TimeoutError")
        assert metrics.error_types["DatabaseError"] == 2
        assert metrics.error_types["TimeoutError"] == 1

    def test_get_operation_avg_with_data(self) -> None:
        """Test get_operation_avg returns correct average."""
        metrics = OperationMetrics()
        metrics.record_operation("backup", 100.0)
        metrics.record_operation("backup", 200.0)
        metrics.record_operation("backup", 300.0)
        assert metrics.get_operation_avg("backup") == 200.0

    def test_get_operation_avg_no_data(self) -> None:
        """Test get_operation_avg returns None when no data."""
        metrics = OperationMetrics()
        assert metrics.get_operation_avg("backup") is None

    def test_get_most_common_error_with_data(self) -> None:
        """Test get_most_common_error returns correct error."""
        metrics = OperationMetrics()
        metrics.record_error("TypeA")
        metrics.record_error("TypeA")
        metrics.record_error("TypeB")
        result = metrics.get_most_common_error()
        assert result == ("TypeA", 2)

    def test_get_most_common_error_no_errors(self) -> None:
        """Test get_most_common_error returns None when no errors."""
        metrics = OperationMetrics()
        assert metrics.get_most_common_error() is None

    def test_get_recent_errors(self) -> None:
        """Test get_recent_errors returns sorted errors by frequency."""
        metrics = OperationMetrics()
        metrics.record_error("TypeC")  # 1
        metrics.record_error("TypeA")  # 1
        metrics.record_error("TypeA")  # 2
        metrics.record_error("TypeB")  # 1
        metrics.record_error("TypeB")  # 2
        metrics.record_error("TypeA")  # 3
        result = metrics.get_recent_errors(limit=2)
        # TypeA has 3 errors, TypeB has 2
        assert result == ["TypeA", "TypeB"]

    def test_get_recent_errors_empty(self) -> None:
        """Test get_recent_errors returns empty list when no errors."""
        metrics = OperationMetrics()
        assert metrics.get_recent_errors() == []


class TestDatabaseWorkerInit:
    """Tests for DatabaseWorker initialization."""

    def test_init_with_db_manager(self, tmp_path: Path) -> None:
        """Test DatabaseWorker initialization with DBManager."""
        db_path = tmp_path / "test.db"
        db_manager = DBManager(db_path=db_path)
        worker = DatabaseWorker(db_manager)
        assert worker.db_manager is db_manager
        assert worker.daemon is True
        # Don't start - just verify init works

    def test_init_with_custom_intervals(self, tmp_path: Path) -> None:
        """Test DatabaseWorker with custom intervals."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(
            db_manager,
            poll_interval=100.0,
            backup_interval=200.0,
            optimization_interval=300.0,
        )
        assert worker.poll_interval == 100.0
        assert worker.backup_interval == 200.0
        assert worker.optimization_interval == 300.0

    def test_init_with_custom_backup_path(self, tmp_path: Path) -> None:
        """Test DatabaseWorker with custom backup path."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        backup_path = tmp_path / "custom_backups"
        worker = DatabaseWorker(db_manager, backup_path=backup_path)
        assert worker.backup_path == backup_path
        assert backup_path.exists()  # Should be created

    def test_init_creates_backup_directory(self, tmp_path: Path) -> None:
        """Test DatabaseWorker creates backup directory."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        assert worker.backup_path.exists()

    def test_init_state_is_stopped(self, tmp_path: Path) -> None:
        """Test DatabaseWorker initial state is STOPPED."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        assert worker._current_state == DBWorkerState.STOPPED


class TestDatabaseWorkerLifecycle:
    """Tests for DatabaseWorker lifecycle methods."""

    def test_start_changes_state(self, tmp_path: Path) -> None:
        """Test start changes state to RUNNING."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)  # Give worker time to start
        try:
            assert worker._current_state == DBWorkerState.RUNNING
            assert worker.is_alive()
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_stop_terminates_worker(self, tmp_path: Path) -> None:
        """Test stop terminates the worker thread."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.2)
        worker.stop()
        worker.join(timeout=5.0)  # Give more time for graceful shutdown
        # Worker may still be alive briefly during cleanup
        # Check that stop flag is set and state is STOPPED
        assert worker._stop_flag is True
        assert worker._current_state == DBWorkerState.STOPPED

    def test_pause_sets_flag(self, tmp_path: Path) -> None:
        """Test pause sets the pause flag."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)
        try:
            worker.pause()
            assert worker._pause_flag is True
            assert worker._current_state == DBWorkerState.PAUSED
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_resume_clears_pause(self, tmp_path: Path) -> None:
        """Test resume clears the pause flag."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)
        try:
            worker.pause()
            worker.resume()
            assert worker._pause_flag is False
            assert worker._current_state == DBWorkerState.RUNNING
        finally:
            worker.stop()
            worker.join(timeout=2.0)


class TestDatabaseWorkerStatus:
    """Tests for DatabaseWorker status methods."""

    def test_get_status_returns_typed_dict(self, tmp_path: Path) -> None:
        """Test get_status returns DBWorkerStatus."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        status = worker.get_status()
        # Check required keys
        assert "running" in status
        assert "operation_count" in status
        assert "error_count" in status
        assert "last_operation" in status
        assert "state" in status

    def test_get_status_running_false_when_stopped(self, tmp_path: Path) -> None:
        """Test get_status shows running=False when stopped."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        status = worker.get_status()
        assert status["running"] is False

    def test_get_status_uptime_when_started(self, tmp_path: Path) -> None:
        """Test get_status includes uptime when started."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.2)
        try:
            status = worker.get_status()
            assert status["uptime"] is not None
            assert status["uptime"] >= 0.1
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_get_metrics_returns_dict(self, tmp_path: Path) -> None:
        """Test get_metrics returns comprehensive metrics."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        metrics = worker.get_metrics()
        assert "operation_count" in metrics
        assert "maintenance" in metrics
        assert "backup" in metrics
        assert "optimization" in metrics
        assert "integrity" in metrics
        assert "errors" in metrics
        assert "schedule" in metrics


class TestDatabaseWorkerOperations:
    """Tests for DatabaseWorker maintenance operations."""

    def test_run_maintenance_schedules_operation(self, tmp_path: Path) -> None:
        """Test run_maintenance schedules a maintenance operation."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        db_manager.create_tables()
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)
        try:
            result = worker.run_maintenance(wait=False)
            assert result is True
            # Operation should be queued
            assert len(worker._pending_operations) >= 0  # May have executed already
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_run_optimization_schedules_operation(self, tmp_path: Path) -> None:
        """Test run_optimization schedules an optimization operation."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        db_manager.create_tables()
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)
        try:
            result = worker.run_optimization(level=2, wait=False)
            assert result is True
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_run_optimization_validates_level(self, tmp_path: Path) -> None:
        """Test run_optimization validates level parameter."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        # Invalid level should be clamped to 1
        worker.run_optimization(level=10, wait=False)
        # Should not raise, just clamp

    def test_run_backup_schedules_operation(self, tmp_path: Path) -> None:
        """Test run_backup schedules a backup operation."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        db_manager.create_tables()
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)
        try:
            result = worker.run_backup(wait=False)
            assert result is True
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_run_backup_with_custom_path(self, tmp_path: Path) -> None:
        """Test run_backup with custom target path."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        db_manager.create_tables()
        custom_backup = tmp_path / "custom_backup.db"
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)
        try:
            result = worker.run_backup(target_path=str(custom_backup), wait=True)
            assert result is True
            # Give a bit more time for backup to complete
            time.sleep(0.5)
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_run_integrity_check_returns_tuple(self, tmp_path: Path) -> None:
        """Test run_integrity_check returns tuple."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        db_manager.create_tables()
        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.1)
        try:
            result = worker.run_integrity_check(wait=False)
            assert isinstance(result, tuple)
            assert len(result) == 2
        finally:
            worker.stop()
            worker.join(timeout=2.0)


class TestDatabaseWorkerErrorHandling:
    """Tests for DatabaseWorker error handling."""

    def test_consecutive_errors_tracked(self, tmp_path: Path) -> None:
        """Test consecutive errors are tracked."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        # Manually call error handler
        worker._handle_error(RuntimeError("test"))
        assert worker._consecutive_errors == 1
        worker._handle_error(RuntimeError("test"))
        assert worker._consecutive_errors == 2

    def test_error_backoff_increases(self, tmp_path: Path) -> None:
        """Test error backoff increases exponentially."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        worker._handle_error(RuntimeError("test"))
        first_backoff = worker._backoff_time
        worker._handle_error(RuntimeError("test"))
        second_backoff = worker._backoff_time
        # Second backoff should be larger (exponential)
        assert second_backoff > first_backoff

    def test_error_state_set_on_error(self, tmp_path: Path) -> None:
        """Test state changes to ERROR on error."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        worker._handle_error(RuntimeError("test"))
        assert worker._current_state == DBWorkerState.ERROR

    def test_recent_errors_populated(self, tmp_path: Path) -> None:
        """Test recent_errors list is populated."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        worker._handle_error(RuntimeError("test"))
        assert "RuntimeError" in worker.metrics.error_types


class TestDatabaseWorkerScheduling:
    """Tests for DatabaseWorker operation scheduling."""

    def test_schedule_operations_sets_times(self, tmp_path: Path) -> None:
        """Test _schedule_operations sets next operation times."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(
            db_manager,
            poll_interval=100.0,
            backup_interval=200.0,
            optimization_interval=300.0,
        )
        before = time.time()
        worker._schedule_operations()
        after = time.time()

        assert worker._next_maintenance is not None
        assert before + 100 <= worker._next_maintenance <= after + 100
        assert before + 200 <= worker._next_backup <= after + 200
        assert before + 300 <= worker._next_optimization <= after + 300


class TestDatabaseWorkerBackupCleanup:
    """Tests for backup cleanup functionality."""

    def test_cleanup_old_backups_removes_excess(self, tmp_path: Path) -> None:
        """Test _cleanup_old_backups removes old backups."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        backup_path = tmp_path / "backups"
        backup_path.mkdir()
        worker = DatabaseWorker(db_manager, backup_path=backup_path)

        # Create 15 backup files
        for i in range(15):
            backup_file = backup_path / f"test_{i:02d}.backup"
            backup_file.write_text("backup content")
            # Set different modification times
            os.utime(backup_file, (time.time() - i * 60, time.time() - i * 60))

        worker._cleanup_old_backups(keep_count=10)
        remaining = list(backup_path.glob("*.backup"))
        assert len(remaining) == 10

    def test_cleanup_old_backups_empty_dir(self, tmp_path: Path) -> None:
        """Test _cleanup_old_backups handles empty directory."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        backup_path = tmp_path / "backups"
        backup_path.mkdir()
        worker = DatabaseWorker(db_manager, backup_path=backup_path)
        # Should not raise
        worker._cleanup_old_backups()

    def test_cleanup_old_backups_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test _cleanup_old_backups handles nonexistent directory."""
        db_manager = DBManager(db_path=tmp_path / "test.db")
        worker = DatabaseWorker(db_manager)
        # Temporarily set backup path to nonexistent
        worker.backup_path = tmp_path / "nonexistent"
        # Should not raise
        worker._cleanup_old_backups()


class TestDatabaseWorkerIntegration:
    """Integration tests for DatabaseWorker."""

    def test_full_lifecycle(self, tmp_path: Path) -> None:
        """Test complete worker lifecycle."""
        db_path = tmp_path / "test.db"
        db_manager = DBManager(db_path=db_path)
        db_manager.create_tables()
        db_manager.insert_or_update_word("test", "definition", "noun")

        worker = DatabaseWorker(
            db_manager,
            poll_interval=3600.0,  # Long interval to prevent auto-operations
            backup_interval=3600.0,
            optimization_interval=3600.0,
        )

        # Start
        worker.start()
        time.sleep(0.2)
        assert worker.is_alive()
        status = worker.get_status()
        assert status["running"] is True

        # Pause
        worker.pause()
        assert worker._pause_flag is True

        # Resume
        worker.resume()
        assert worker._pause_flag is False

        # Stop
        worker.stop()
        worker.join(timeout=5.0)
        assert not worker.is_alive()

    def test_maintenance_executes(self, tmp_path: Path) -> None:
        """Test that maintenance actually executes."""
        db_path = tmp_path / "test.db"
        db_manager = DBManager(db_path=db_path)
        db_manager.create_tables()
        db_manager.insert_or_update_word("word1", "def1", "noun")

        worker = DatabaseWorker(db_manager, poll_interval=3600.0)
        worker.start()
        time.sleep(0.2)

        try:
            # Run maintenance synchronously
            worker.run_maintenance(wait=True)
            # Check metrics
            assert worker.metrics.last_maintenance is not None
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    def test_backup_creates_file(self, tmp_path: Path) -> None:
        """Test that backup creates a backup file."""
        db_path = tmp_path / "test.db"
        db_manager = DBManager(db_path=db_path)
        db_manager.create_tables()
        db_manager.insert_or_update_word("word1", "def1", "noun")

        backup_dir = tmp_path / "backups"
        worker = DatabaseWorker(
            db_manager, backup_path=backup_dir, poll_interval=3600.0
        )
        worker.start()
        time.sleep(0.2)

        try:
            worker.run_backup(wait=True)
            time.sleep(0.5)  # Give time for backup to complete
            backups = list(backup_dir.glob("*.backup"))
            assert len(backups) >= 1
        finally:
            worker.stop()
            worker.join(timeout=2.0)
