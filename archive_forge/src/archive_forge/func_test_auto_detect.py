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
def test_auto_detect(self):
    term_width, term_height = get_terminal_size()
    fac = 1.05
    cols = range(int(term_width * fac))
    index = range(10)
    df = DataFrame(index=index, columns=cols)
    with option_context('mode.sim_interactive', True):
        with option_context('display.max_rows', None):
            with option_context('display.max_columns', None):
                assert has_expanded_repr(df)
        with option_context('display.max_rows', 0):
            with option_context('display.max_columns', 0):
                assert has_horizontally_truncated_repr(df)
        index = range(int(term_height * fac))
        df = DataFrame(index=index, columns=cols)
        with option_context('display.max_rows', 0):
            with option_context('display.max_columns', None):
                assert has_expanded_repr(df)
                assert has_vertically_truncated_repr(df)
        with option_context('display.max_rows', None):
            with option_context('display.max_columns', 0):
                assert has_horizontally_truncated_repr(df)