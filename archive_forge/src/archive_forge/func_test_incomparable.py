import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def test_incomparable(self):
    h = [5, 4, 'a', 2, 1, 0]
    pytest.raises(TypeError, MappedQueue, h)