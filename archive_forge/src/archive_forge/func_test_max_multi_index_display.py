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
def test_max_multi_index_display(self):
    arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'], ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
    tuples = list(zip(*arrays))
    index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
    s = Series(np.random.default_rng(2).standard_normal(8), index=index)
    with option_context('display.max_rows', 10):
        assert len(str(s).split('\n')) == 10
    with option_context('display.max_rows', 3):
        assert len(str(s).split('\n')) == 5
    with option_context('display.max_rows', 2):
        assert len(str(s).split('\n')) == 5
    with option_context('display.max_rows', 1):
        assert len(str(s).split('\n')) == 4
    with option_context('display.max_rows', 0):
        assert len(str(s).split('\n')) == 10
    s = Series(np.random.default_rng(2).standard_normal(8), None)
    with option_context('display.max_rows', 10):
        assert len(str(s).split('\n')) == 9
    with option_context('display.max_rows', 3):
        assert len(str(s).split('\n')) == 4
    with option_context('display.max_rows', 2):
        assert len(str(s).split('\n')) == 4
    with option_context('display.max_rows', 1):
        assert len(str(s).split('\n')) == 3
    with option_context('display.max_rows', 0):
        assert len(str(s).split('\n')) == 9