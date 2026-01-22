import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_update_root(self):
    h = [0, 20, 10, 60, 30, 50, 40]
    h_updated = [10, 20, 35, 60, 30, 50, 40]
    q = self._make_mapped_queue(h)
    removed = q.update(0, 35, priority=35)
    assert q.heap == h_updated