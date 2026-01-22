import pytest
from mpmath import *
def test_double_7():
    assert ae(quadts(lambda x, y: exp(-x * x - y * y), [-inf, inf], [-inf, inf]), pi)