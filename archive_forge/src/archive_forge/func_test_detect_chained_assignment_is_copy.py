from string import ascii_letters
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_detect_chained_assignment_is_copy(self):
    df = DataFrame({'a': [1]}).dropna()
    assert df._is_copy is None
    df['a'] += 1