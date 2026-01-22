import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_print_none_width(self):
    a = Series(Categorical([1, 2, 3, 4]))
    exp = '0    1\n1    2\n2    3\n3    4\ndtype: category\nCategories (4, int64): [1, 2, 3, 4]'
    with option_context('display.width', None):
        assert exp == repr(a)