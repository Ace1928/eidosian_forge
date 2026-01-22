import numpy as np
import pytest
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_view_index(self, simple_index):
    index = simple_index
    msg = 'Passing a type in RangeIndex.view is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        index.view(Index)