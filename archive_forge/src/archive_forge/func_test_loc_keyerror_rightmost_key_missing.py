import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_keyerror_rightmost_key_missing():
    df = DataFrame({'A': [100, 100, 200, 200, 300, 300], 'B': [10, 10, 20, 21, 31, 33], 'C': range(6)})
    df = df.set_index(['A', 'B'])
    with pytest.raises(KeyError, match='^1$'):
        df.loc[100, 1]