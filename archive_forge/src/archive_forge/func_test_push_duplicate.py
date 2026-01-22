import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_push_duplicate(self):
    to_push = [2, 1, 0]
    h_sifted = [0, 2, 1]
    q = MappedQueue()
    for elt in to_push:
        inserted = q.push(elt, priority=elt)
        assert inserted
    assert q.heap == h_sifted
    self._check_map(q)
    inserted = q.push(1, priority=1)
    assert not inserted