from datetime import datetime
from io import StringIO
import itertools
import re
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_html_wide_multiindex_cols(self):
    max_cols = 20
    mcols = MultiIndex.from_product([np.arange(max_cols // 2), ['foo', 'bar']], names=['first', 'second'])
    df = DataFrame([['a' * 25] * len(mcols)] * 10, columns=mcols)
    reg_repr = df._repr_html_()
    assert '...' not in reg_repr
    mcols = MultiIndex.from_product((np.arange(1 + max_cols // 2), ['foo', 'bar']), names=['first', 'second'])
    df = DataFrame([['a' * 25] * len(mcols)] * 10, columns=mcols)
    with option_context('display.max_rows', 60, 'display.max_columns', 20):
        assert '...' in df._repr_html_()