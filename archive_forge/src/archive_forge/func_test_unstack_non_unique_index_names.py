from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_unstack_non_unique_index_names(self, future_stack):
    idx = MultiIndex.from_tuples([('a', 'b'), ('c', 'd')], names=['c1', 'c1'])
    df = DataFrame([1, 2], index=idx)
    msg = 'The name c1 occurs multiple times, use a level number'
    with pytest.raises(ValueError, match=msg):
        df.unstack('c1')
    with pytest.raises(ValueError, match=msg):
        df.T.stack('c1', future_stack=future_stack)