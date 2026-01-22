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
def test_pass_function(self, data):
    result = data.pivot_table('D', index=lambda x: x // 5, columns=data.C)
    expected = data.pivot_table('D', index=data.index // 5, columns='C')
    tm.assert_frame_equal(result, expected)