import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_empty_print(self):
    factor = Categorical([], Index(['a', 'b', 'c'], dtype=object))
    expected = "[], Categories (3, object): ['a', 'b', 'c']"
    actual = repr(factor)
    assert actual == expected
    assert expected == actual
    factor = Categorical([], Index(['a', 'b', 'c'], dtype=object), ordered=True)
    expected = "[], Categories (3, object): ['a' < 'b' < 'c']"
    actual = repr(factor)
    assert expected == actual
    factor = Categorical([], [])
    expected = '[], Categories (0, object): []'
    assert expected == repr(factor)