import pytest
from mpmath import *
def test_double_1():
    assert ae(quadts(lambda x, y: cos(x + y / 2), [-pi / 2, pi / 2], [0, pi]), 4)