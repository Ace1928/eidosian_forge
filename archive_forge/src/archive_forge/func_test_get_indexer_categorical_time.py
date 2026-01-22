from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_categorical_time(self):
    midx = MultiIndex.from_product([Categorical(['a', 'b', 'c']), Categorical(date_range('2012-01-01', periods=3, freq='h'))])
    result = midx.get_indexer(midx)
    tm.assert_numpy_array_equal(result, np.arange(9, dtype=np.intp))