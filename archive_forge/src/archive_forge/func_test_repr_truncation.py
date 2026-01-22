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
def test_repr_truncation(self):
    max_len = 20
    with option_context('display.max_colwidth', max_len):
        df = DataFrame({'A': np.random.default_rng(2).standard_normal(10), 'B': ['a' * np.random.default_rng(2).integers(max_len - 1, max_len + 1) for _ in range(10)]})
        r = repr(df)
        r = r[r.find('\n') + 1:]
        adj = printing.get_adjustment()
        for line, value in zip(r.split('\n'), df['B']):
            if adj.len(value) + 1 > max_len:
                assert '...' in line
            else:
                assert '...' not in line
    with option_context('display.max_colwidth', 999999):
        assert '...' not in repr(df)
    with option_context('display.max_colwidth', max_len + 2):
        assert '...' not in repr(df)