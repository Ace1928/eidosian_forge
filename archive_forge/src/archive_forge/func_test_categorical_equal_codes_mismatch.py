import pytest
from pandas import Categorical
import pandas._testing as tm
def test_categorical_equal_codes_mismatch():
    categories = [1, 2, 3, 4]
    msg = 'Categorical\\.codes are different\n\nCategorical\\.codes values are different \\(50\\.0 %\\)\n\\[left\\]:  \\[0, 1, 3, 2\\]\n\\[right\\]: \\[0, 1, 2, 3\\]'
    c1 = Categorical([1, 2, 4, 3], categories=categories)
    c2 = Categorical([1, 2, 3, 4], categories=categories)
    with pytest.raises(AssertionError, match=msg):
        tm.assert_categorical_equal(c1, c2)