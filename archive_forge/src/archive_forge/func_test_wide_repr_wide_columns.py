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
def test_wide_repr_wide_columns(self):
    with option_context('mode.sim_interactive', True, 'display.max_columns', 20):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)), columns=['a' * 90, 'b' * 90, 'c' * 90])
        rep_str = repr(df)
        assert len(rep_str.splitlines()) == 20