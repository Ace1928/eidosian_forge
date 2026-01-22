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
def test_unstack_single_index_series():
    msg = 'index must be a MultiIndex to unstack.*'
    with pytest.raises(ValueError, match=msg):
        Series(dtype=np.int64).unstack()