from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_margins_no_values_two_rows(self, data):
    result = data[['A', 'B', 'C']].pivot_table(index=['A', 'B'], columns='C', aggfunc=len, margins=True)
    assert result.All.tolist() == [3.0, 1.0, 4.0, 3.0, 11.0]