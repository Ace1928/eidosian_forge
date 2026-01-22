import pytest
from pandas import Categorical
import pandas._testing as tm
def test_categorical_equal_categories_mismatch():
    msg = "Categorical\\.categories are different\n\nCategorical\\.categories values are different \\(25\\.0 %\\)\n\\[left\\]:  Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)\n\\[right\\]: Index\\(\\[1, 2, 3, 5\\], dtype='int64'\\)"
    c1 = Categorical([1, 2, 3, 4])
    c2 = Categorical([1, 2, 3, 5])
    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)