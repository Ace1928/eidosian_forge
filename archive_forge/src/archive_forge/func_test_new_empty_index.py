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
def test_new_empty_index(self):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
    df1.index.name = 'foo'
    assert df2.index.name is None