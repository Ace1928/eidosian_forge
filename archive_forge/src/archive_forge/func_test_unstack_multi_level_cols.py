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
def test_unstack_multi_level_cols(self):
    df = DataFrame([[0.0, 0.0], [0.0, 0.0]], columns=MultiIndex.from_tuples([['B', 'C'], ['B', 'D']], names=['c1', 'c2']), index=MultiIndex.from_tuples([[10, 20, 30], [10, 20, 40]], names=['i1', 'i2', 'i3']))
    assert df.unstack(['i2', 'i1']).columns.names[-2:] == ['i2', 'i1']