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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='surrogates not allowed')
def test_column_name_contains_unicode_surrogate(self):
    colname = '\ud83d'
    df = DataFrame({colname: []})
    assert colname not in dir(df)
    assert df.columns[0] == colname