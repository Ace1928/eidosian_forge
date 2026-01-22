from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('length, max_rows, min_rows, expected', [(10, 10, 10, 10), (10, 10, None, 10), (10, 8, None, 8), (20, 30, 10, 30), (50, 30, 10, 10), (100, 60, 10, 10), (60, 60, 10, 60), (61, 60, 10, 10)])
def test_max_rows_fitted(self, length, min_rows, max_rows, expected):
    """Check that display logic is correct.

        GH #37359

        See description here:
        https://pandas.pydata.org/docs/dev/user_guide/options.html#frequently-used-options
        """
    formatter = fmt.DataFrameFormatter(DataFrame(np.random.default_rng(2).random((length, 3))), max_rows=max_rows, min_rows=min_rows)
    result = formatter.max_rows_fitted
    assert result == expected