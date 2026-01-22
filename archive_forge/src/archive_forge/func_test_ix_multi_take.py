from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ix_multi_take(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
    rs = df.loc[df.index == 0, :]
    xp = df.reindex([0])
    tm.assert_frame_equal(rs, xp)
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 2)))
    rs = df.loc[df.index == 0, df.columns == 1]
    xp = df.reindex(index=[0], columns=[1])
    tm.assert_frame_equal(rs, xp)