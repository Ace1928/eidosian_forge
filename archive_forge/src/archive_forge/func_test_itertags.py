import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_itertags(self):
    tl = rlc.TaggedList((1, 2, 3), tags=('a', 'b', 'c'))
    assert tuple(tl.itertags()) == ('a', 'b', 'c')