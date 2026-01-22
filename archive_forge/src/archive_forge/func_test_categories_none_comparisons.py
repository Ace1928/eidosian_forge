import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categories_none_comparisons(self):
    factor = Categorical(['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'], ordered=True)
    tm.assert_categorical_equal(factor, factor)