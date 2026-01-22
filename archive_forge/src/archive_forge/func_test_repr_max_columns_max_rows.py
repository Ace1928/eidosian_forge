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
def test_repr_max_columns_max_rows(self):
    term_width, term_height = get_terminal_size()
    if term_width < 10 or term_height < 10:
        pytest.skip(f'terminal size too small, {term_width} x {term_height}')

    def mkframe(n):
        index = [f'{i:05d}' for i in range(n)]
        return DataFrame(0, index, index)
    df6 = mkframe(6)
    df10 = mkframe(10)
    with option_context('mode.sim_interactive', True):
        with option_context('display.width', term_width * 2):
            with option_context('display.max_rows', 5, 'display.max_columns', 5):
                assert not has_expanded_repr(mkframe(4))
                assert not has_expanded_repr(mkframe(5))
                assert not has_expanded_repr(df6)
                assert has_doubly_truncated_repr(df6)
            with option_context('display.max_rows', 20, 'display.max_columns', 10):
                assert not has_expanded_repr(df6)
                assert not has_truncated_repr(df6)
            with option_context('display.max_rows', 9, 'display.max_columns', 10):
                assert not has_expanded_repr(df10)
                assert has_vertically_truncated_repr(df10)
        with option_context('display.max_columns', 100, 'display.max_rows', term_width * 20, 'display.width', None):
            df = mkframe(term_width // 7 - 2)
            assert not has_expanded_repr(df)
            df = mkframe(term_width // 7 + 2)
            printing.pprint_thing(df._repr_fits_horizontal_())
            assert has_expanded_repr(df)