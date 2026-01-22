from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rdiv_ndarray_0d(self):
    td = Timedelta(10, unit='d')
    arr = np.array(td.asm8)
    assert arr / td == 1