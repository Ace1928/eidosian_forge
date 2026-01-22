from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
@td.skip_array_manager_invalid_test
def test_setitem_column_update_inplace(self, using_copy_on_write, warn_copy_on_write):
    labels = [f'c{i}' for i in range(10)]
    df = DataFrame({col: np.zeros(len(labels)) for col in labels}, index=labels)
    values = df._mgr.blocks[0].values
    with tm.raises_chained_assignment_error():
        for label in df.columns:
            df[label][label] = 1
    if not using_copy_on_write:
        assert np.all(values[np.arange(10), np.arange(10)] == 1)
    else:
        assert np.all(values[np.arange(10), np.arange(10)] == 0)