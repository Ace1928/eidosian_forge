import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test__imul__(self):
    tl = rlc.TaggedList((1, 2), tags=('a', 'b'))
    tl *= 3
    assert len(tl) == 6
    assert tl.tags == ('a', 'b', 'a', 'b', 'a', 'b')
    assert tuple(tl) == (1, 2, 1, 2, 1, 2)