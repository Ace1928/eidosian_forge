from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('value', [''.join(elements) for repetition in (1, 2) for elements in product('+-, ', repeat=repetition)])
def test_string_without_numbers(value):
    msg = 'symbols w/o a number' if value != '--' else 'only leading negative signs are allowed'
    with pytest.raises(ValueError, match=msg):
        Timedelta(value)