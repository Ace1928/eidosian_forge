import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_non_unique_idvars(self):
    df = DataFrame({'A_A1': [1, 2, 3, 4, 5], 'B_B1': [1, 2, 3, 4, 5], 'x': [1, 1, 1, 1, 1]})
    msg = 'the id variables need to uniquely identify each row'
    with pytest.raises(ValueError, match=msg):
        wide_to_long(df, ['A_A', 'B_B'], i='x', j='colname')