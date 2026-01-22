import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
def test_positional_take(self, ordered):
    cat = Categorical(['a', 'a', 'b', 'b'], categories=['b', 'a'], ordered=ordered)
    result = cat.take([0, 1, 2], allow_fill=False)
    expected = Categorical(['a', 'a', 'b'], categories=cat.categories, ordered=ordered)
    tm.assert_categorical_equal(result, expected)