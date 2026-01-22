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
def test_wide_repr_wide_long_columns(self):
    with option_context('mode.sim_interactive', True):
        df = DataFrame({'a': ['a' * 30, 'b' * 30], 'b': ['c' * 70, 'd' * 80]})
        result = repr(df)
        assert 'ccccc' in result
        assert 'ddddd' in result