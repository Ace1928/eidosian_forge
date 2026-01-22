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
@pytest.mark.parametrize('max_rows,min_rows,expected', [(10, 4, 'html_repr_max_rows_10_min_rows_4'), (12, None, 'html_repr_max_rows_12_min_rows_None'), (10, 12, 'html_repr_max_rows_10_min_rows_12'), (None, 12, 'html_repr_max_rows_None_min_rows_12')])
def test_html_repr_min_rows(self, datapath, max_rows, min_rows, expected):
    df = DataFrame({'a': range(61)})
    expected = expected_html(datapath, expected)
    with option_context('display.max_rows', max_rows, 'display.min_rows', min_rows):
        result = df._repr_html_()
    assert result == expected