import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
@pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
def test_validate_inplace_raises(self, value):
    cat = Categorical(['A', 'B', 'B', 'C', 'A'])
    msg = f'For argument "inplace" expected type bool, received type {type(value).__name__}'
    with pytest.raises(ValueError, match=msg):
        cat.sort_values(inplace=value)