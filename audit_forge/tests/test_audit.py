import pytest
from pathlib import Path
import os
import shutil
from audit_forge.coverage import CoverageTracker
from audit_forge.tasks import IdempotentTaskManager

@pytest.fixture
def temp_workspace(tmp_path):
    # Setup a mock home-like structure
    todo = tmp_path / "TODO.md"
    todo.write_text("# TODO\n\n## Immediate\n- [ ] Task 1\n")
    return tmp_path

def test_coverage_tracker(temp_workspace):
    persistence = temp_workspace / "coverage.json"
    tracker = CoverageTracker(persistence)
    tracker.mark_reviewed("scripts/test.py", "agent-001")
    
    assert "scripts/test.py" in tracker.data["coverage"]
    assert tracker.data["coverage"]["scripts/test.py"]["agent"] == "agent-001"
    
    # Test persistence
    new_tracker = CoverageTracker(persistence)
    assert "scripts/test.py" in new_tracker.data["coverage"]

def test_task_manager(temp_workspace):
    todo_path = temp_workspace / "TODO.md"
    manager = IdempotentTaskManager(todo_path)
    
    # Test add task
    manager.add_task("Immediate", "New Task", "ID-01")
    content = todo_path.read_text()
    assert "- [ ] **ID-01** New Task" in content
    
    # Test idempotency
    assert manager.add_task("Immediate", "New Task", "ID-01") == False
    
    # Test cross off
    manager.cross_off_task("ID-01")
    content = todo_path.read_text()
    assert "- [x] **ID-01** New Task" in content

def test_coverage_gap_analysis(temp_workspace):
    # Create some dummy files
    scripts_dir = temp_workspace / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "a.py").write_text("")
    (scripts_dir / "b.py").write_text("")
    
    tracker = CoverageTracker(temp_workspace / "coverage.json")
    tracker.mark_reviewed("scripts/a.py", "agent-001")
    
    unreviewed = tracker.get_unreviewed_files(temp_workspace)
    assert "scripts/b.py" in unreviewed
    assert "scripts/a.py" not in unreviewed
