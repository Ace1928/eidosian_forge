import pytest
from mpmath import *
def test_double_4():
    assert ae(quadts(lambda x, y: 1 / (1 - x * x * y * y), [0, 1], [0, 1]), pi ** 2 / 8)