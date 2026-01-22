import numpy as np
import pytest
from pandas import (
def test_is_strictly_monotonic_decreasing():
    idx = MultiIndex(levels=[['baz', 'bar'], ['next', 'mom']], codes=[[0, 0, 1, 1], [0, 0, 0, 1]])
    assert idx.is_monotonic_decreasing is True
    assert idx._is_strictly_monotonic_decreasing is False