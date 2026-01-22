import pytest
import pickle
from io import BytesIO
import rpy2.rlike.container as rlc
def test_getsetitemwithnone(self):
    x = rlc.OrdDict()
    x['a'] = 1
    x[None] = 2
    assert len(x) == 2
    x['b'] = 5
    assert len(x) == 3
    assert x['a'] == 1
    assert x['b'] == 5
    assert x.index('a') == 0
    assert x.index('b') == 2