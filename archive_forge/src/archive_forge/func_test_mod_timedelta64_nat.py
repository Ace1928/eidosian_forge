from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_mod_timedelta64_nat(self):
    td = Timedelta(hours=37)
    result = td % np.timedelta64('NaT', 'ns')
    assert result is NaT