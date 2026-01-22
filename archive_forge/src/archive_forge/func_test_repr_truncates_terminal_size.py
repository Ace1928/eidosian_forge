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
def test_repr_truncates_terminal_size(self, monkeypatch):
    terminal_size = (118, 96)
    monkeypatch.setattr('pandas.io.formats.format.get_terminal_size', lambda: terminal_size)
    index = range(5)
    columns = MultiIndex.from_tuples([('This is a long title with > 37 chars.', 'cat'), ('This is a loooooonger title with > 43 chars.', 'dog')])
    df = DataFrame(1, index=index, columns=columns)
    result = repr(df)
    h1, h2 = result.split('\n')[:2]
    assert 'long' in h1
    assert 'loooooonger' in h1
    assert 'cat' in h2
    assert 'dog' in h2
    df2 = DataFrame({'A' * 41: [1, 2], 'B' * 41: [1, 2]})
    result = repr(df2)
    assert df2.columns[0] in result.split('\n')[0]