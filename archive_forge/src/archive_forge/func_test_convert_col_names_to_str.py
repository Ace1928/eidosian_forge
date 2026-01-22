import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_convert_col_names_to_str():
    df = pd.DataFrame(np.random.random((10, 2)))
    assert all((not isinstance(col, str) for col in df.columns))
    df = _convert_col_names_to_str(df)
    assert all((isinstance(col, str) for col in df.columns))