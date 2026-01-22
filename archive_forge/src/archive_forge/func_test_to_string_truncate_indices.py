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
@pytest.mark.parametrize('index_scalar', ['a' * 10, 1, Timestamp(2020, 1, 1), pd.Period('2020-01-01')])
@pytest.mark.parametrize('h', [10, 20])
@pytest.mark.parametrize('w', [10, 20])
def test_to_string_truncate_indices(self, index_scalar, h, w):
    with option_context('display.expand_frame_repr', False):
        df = DataFrame(index=[index_scalar] * h, columns=[str(i) * 10 for i in range(w)])
        with option_context('display.max_rows', 15):
            if h == 20:
                assert has_vertically_truncated_repr(df)
            else:
                assert not has_vertically_truncated_repr(df)
        with option_context('display.max_columns', 15):
            if w == 20:
                assert has_horizontally_truncated_repr(df)
            else:
                assert not has_horizontally_truncated_repr(df)
        with option_context('display.max_rows', 15, 'display.max_columns', 15):
            if h == 20 and w == 20:
                assert has_doubly_truncated_repr(df)
            else:
                assert not has_doubly_truncated_repr(df)