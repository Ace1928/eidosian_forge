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
def test_logical_compat(self, simple_index):
    idx = simple_index
    with pytest.raises(TypeError, match='cannot perform all'):
        idx.all()
    with pytest.raises(TypeError, match='cannot perform any'):
        idx.any()