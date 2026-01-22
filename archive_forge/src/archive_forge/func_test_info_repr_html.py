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
def test_info_repr_html(self):
    max_rows = 60
    max_cols = 20
    h, w = (max_rows + 1, max_cols - 1)
    df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
    assert '&lt;class' not in df._repr_html_()
    with option_context('display.large_repr', 'info'):
        assert '&lt;class' in df._repr_html_()
    h, w = (max_rows - 1, max_cols + 1)
    df = DataFrame({k: np.arange(1, 1 + h) for k in np.arange(w)})
    assert '<class' not in df._repr_html_()
    with option_context('display.large_repr', 'info', 'display.max_columns', max_cols):
        assert '&lt;class' in df._repr_html_()