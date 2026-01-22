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
def test_expand_frame_repr(self):
    df_small = DataFrame('hello', index=[0], columns=[0])
    df_wide = DataFrame('hello', index=[0], columns=range(10))
    df_tall = DataFrame('hello', index=range(30), columns=range(5))
    with option_context('mode.sim_interactive', True):
        with option_context('display.max_columns', 10, 'display.width', 20, 'display.max_rows', 20, 'display.show_dimensions', True):
            with option_context('display.expand_frame_repr', True):
                assert not has_truncated_repr(df_small)
                assert not has_expanded_repr(df_small)
                assert not has_truncated_repr(df_wide)
                assert has_expanded_repr(df_wide)
                assert has_vertically_truncated_repr(df_tall)
                assert has_expanded_repr(df_tall)
            with option_context('display.expand_frame_repr', False):
                assert not has_truncated_repr(df_small)
                assert not has_expanded_repr(df_small)
                assert not has_horizontally_truncated_repr(df_wide)
                assert not has_expanded_repr(df_wide)
                assert has_vertically_truncated_repr(df_tall)
                assert not has_expanded_repr(df_tall)