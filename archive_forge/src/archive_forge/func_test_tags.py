import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_tags(self):
    tn = ['a', 'b', 'c']
    tv = [1, 2, 3]
    tl = rlc.TaggedList(tv, tags=tn)
    tags = tl.tags
    assert isinstance(tags, tuple) is True
    assert tags == ('a', 'b', 'c')
    tn = ['d', 'e', 'f']
    tl.tags = tn
    assert isinstance(tags, tuple) is True
    assert tl.tags == tuple(tn)