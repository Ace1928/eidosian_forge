import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_from_items(self):
    od = rlc.OrdDict((('a', 1), ('b', 2), ('c', 3)))
    tl = rlc.TaggedList.from_items(od)
    assert tl.tags == ('a', 'b', 'c')
    assert tuple(tl) == (1, 2, 3)
    tl = rlc.TaggedList.from_items({'a': 1, 'b': 2, 'c': 3})
    assert set(tl.tags) == set(('a', 'b', 'c'))
    assert set(tuple(tl)) == set((1, 2, 3))