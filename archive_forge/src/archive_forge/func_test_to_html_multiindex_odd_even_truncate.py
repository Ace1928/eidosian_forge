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
@pytest.mark.parametrize('max_rows,expected', [(60, 'gh14882_expected_output_1'), (56, 'gh14882_expected_output_2')])
def test_to_html_multiindex_odd_even_truncate(max_rows, expected, datapath):
    index = MultiIndex.from_product([[100, 200, 300], [10, 20, 30], [1, 2, 3, 4, 5, 6, 7]], names=['a', 'b', 'c'])
    df = DataFrame({'n': range(len(index))}, index=index)
    result = df.to_html(max_rows=max_rows)
    expected = expected_html(datapath, expected)
    assert result == expected