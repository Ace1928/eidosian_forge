from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_holds_integer_deprecated(self, simple_index):
    idx = simple_index
    msg = f'{type(idx).__name__}.holds_integer is deprecated. '
    with tm.assert_produces_warning(FutureWarning, match=msg):
        idx.holds_integer()