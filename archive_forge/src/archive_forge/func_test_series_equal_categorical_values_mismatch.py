import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_equal_categorical_values_mismatch(rtol, using_infer_string):
    if using_infer_string:
        msg = "Series are different\n\nSeries values are different \\(66\\.66667 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\['a', 'b', 'c'\\]\nCategories \\(3, string\\): \\[a, b, c\\]\n\\[right\\]: \\['a', 'c', 'b'\\]\nCategories \\(3, string\\): \\[a, b, c\\]"
    else:
        msg = "Series are different\n\nSeries values are different \\(66\\.66667 %\\)\n\\[index\\]: \\[0, 1, 2\\]\n\\[left\\]:  \\['a', 'b', 'c'\\]\nCategories \\(3, object\\): \\['a', 'b', 'c'\\]\n\\[right\\]: \\['a', 'c', 'b'\\]\nCategories \\(3, object\\): \\['a', 'b', 'c'\\]"
    s1 = Series(Categorical(['a', 'b', 'c']))
    s2 = Series(Categorical(['a', 'c', 'b']))
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)