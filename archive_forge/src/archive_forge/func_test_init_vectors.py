import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
@pytest.mark.parametrize('cls,values', [(robjects.StrVector, ['abc', 'def']), (robjects.IntVector, [123, 456]), (robjects.FloatVector, [123.0, 456.0]), (robjects.BoolVector, [True, False])])
def test_init_vectors(cls, values):
    vec = cls(values)
    assert len(vec) == len(values)
    for x, y in zip(vec, values):
        assert x == y