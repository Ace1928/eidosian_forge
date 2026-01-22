import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_extend(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
    tl.extend([4, 5])
    assert tuple(tl.itertags()) == ('a', 'b', 'c', None, None)
    assert tuple(tl) == (1, 2, 3, 4, 5)