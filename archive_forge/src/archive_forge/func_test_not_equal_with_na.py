import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('categories', [['a', 'b'], [0, 1], [Timestamp('2019'), Timestamp('2020')]])
def test_not_equal_with_na(self, categories):
    c1 = Categorical.from_codes([-1, 0], categories=categories)
    c2 = Categorical.from_codes([0, 1], categories=categories)
    result = c1 != c2
    assert result.all()