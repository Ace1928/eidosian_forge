import numpy as np
import pytest
from pandas import (
def test_equals_string_dtype(self, any_string_dtype):
    idx = CategoricalIndex(list('abc'), name='B')
    other = Index(['a', 'b', 'c'], name='B', dtype=any_string_dtype)
    assert idx.equals(other)