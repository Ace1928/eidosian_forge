import pytest
from pathlib import Path
from audit_forge.tasks import IdempotentTaskManager

def test_task_manager(tmp_path):
    todo_file = tmp_path / "TODO.md"
    todo_file.write_text("# TODO\n\n## Section 1\n- [ ] Task A\n")
    
    manager = IdempotentTaskManager(todo_file)
    
    # Test duplicate detection
    assert not manager.add_task("Section 1", "Task A")
    
    # Test addition
    assert manager.add_task("Section 1", "Task B")
    content = todo_file.read_text()
    assert "- [ ] Task B" in content
    
    # Test new section creation
    assert manager.add_task("New Section", "Task C")
    content = todo_file.read_text()
    assert "## New Section" in content
    assert "- [ ] Task C" in content

def test_cross_off(tmp_path):
    todo_file = tmp_path / "TODO.md"
    todo_file.write_text("# TODO\n- [ ] Task A\n")
    
    manager = IdempotentTaskManager(todo_file)
    assert manager.cross_off_task("Task A")
    
    content = todo_file.read_text()
    assert "- [x] Task A" in content
