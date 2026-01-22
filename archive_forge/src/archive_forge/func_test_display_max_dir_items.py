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
def test_display_max_dir_items(self):
    columns = ['a' + str(i) for i in range(420)]
    values = [range(420), range(420)]
    df = DataFrame(values, columns=columns)
    assert 'a99' in dir(df)
    assert 'a100' not in dir(df)
    with option_context('display.max_dir_items', 300):
        df = DataFrame(values, columns=columns)
        assert 'a299' in dir(df)
        assert 'a300' not in dir(df)
    with option_context('display.max_dir_items', None):
        df = DataFrame(values, columns=columns)
        assert 'a419' in dir(df)