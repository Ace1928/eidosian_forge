import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.queue.queue_manager import QueueManager, TaskPriority


def test_enqueue_and_dequeue_basic():
    qm = QueueManager[str]()
    assert qm.enqueue("task1").unwrap()
    assert len(qm) == 1
    result = qm.dequeue()
    assert result.is_success
    assert result.unwrap() == "task1"
    assert qm.is_empty


def test_enqueue_duplicate():
    qm = QueueManager[str]()
    assert qm.enqueue("a").unwrap()
    assert not qm.enqueue("a").unwrap()
    assert len(qm) == 1


def test_priority_order():
    qm = QueueManager[str]()
    assert qm.enqueue("low", priority=TaskPriority.LOW).unwrap()
    assert qm.enqueue("high", priority=TaskPriority.HIGH).unwrap()
    assert qm.dequeue().unwrap() == "high"
    assert qm.dequeue().unwrap() == "low"
