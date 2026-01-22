import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_validate_ascending_for_value_error(self):
    df = DataFrame({'D': [23, 7, 21]})
    msg = 'For argument "ascending" expected type bool, received type str.'
    with pytest.raises(ValueError, match=msg):
        df.sort_values(by='D', ascending='False')