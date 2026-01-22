import pytest
from mpmath import *
def test_double_6():
    assert ae(quadts(lambda x, y: exp(-(x + y)), [0, inf], [0, inf]), 1)