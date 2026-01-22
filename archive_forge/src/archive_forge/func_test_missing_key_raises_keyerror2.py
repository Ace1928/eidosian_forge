import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_missing_key_raises_keyerror2(self):
    ser = Series(-1, index=MultiIndex.from_product([[0, 1]] * 2))
    with pytest.raises(KeyError, match='\\(0, 3\\)'):
        ser.loc[0, 3]