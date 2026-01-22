from copy import deepcopy
import inspect
import pydoc
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import option_context
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_series_put_names(self, float_string_frame):
    series = float_string_frame._series
    for k, v in series.items():
        assert v.name == k