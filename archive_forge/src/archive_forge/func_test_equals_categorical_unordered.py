import numpy as np
import pytest
from pandas import (
def test_equals_categorical_unordered(self):
    a = CategoricalIndex(['A'], categories=['A', 'B'])
    b = CategoricalIndex(['A'], categories=['B', 'A'])
    c = CategoricalIndex(['C'], categories=['B', 'A'])
    assert a.equals(b)
    assert not a.equals(c)
    assert not b.equals(c)