import pytest
from pandas import Categorical
import pandas._testing as tm
@pytest.mark.parametrize('c', [Categorical([1, 2, 3, 4]), Categorical([1, 2, 3, 4], categories=[1, 2, 3, 4, 5])])
def test_categorical_equal(c):
    tm.assert_categorical_equal(c, c)