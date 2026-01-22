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
def test_getitem_fancy_scalar(self, float_frame):
    f = float_frame
    ix = f.loc
    for col in f.columns:
        ts = f[col]
        for idx in f.index[::5]:
            assert ix[idx, col] == ts[idx]