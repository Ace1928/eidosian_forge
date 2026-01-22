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
def test_str_max_colwidth(self):
    df = DataFrame([{'a': 'foo', 'b': 'bar', 'c': 'uncomfortably long line with lots of stuff', 'd': 1}, {'a': 'foo', 'b': 'bar', 'c': 'stuff', 'd': 1}])
    df.set_index(['a', 'b', 'c'])
    assert str(df) == '     a    b                                           c  d\n0  foo  bar  uncomfortably long line with lots of stuff  1\n1  foo  bar                                       stuff  1'
    with option_context('max_colwidth', 20):
        assert str(df) == '     a    b                    c  d\n0  foo  bar  uncomfortably lo...  1\n1  foo  bar                stuff  1'