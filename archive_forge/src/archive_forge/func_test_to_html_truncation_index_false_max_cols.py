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
@pytest.mark.parametrize('index', [False, 0])
@pytest.mark.parametrize('col_index_named, expected_output', [(False, 'gh22783_expected_output'), (True, 'gh22783_named_columns_index')])
def test_to_html_truncation_index_false_max_cols(datapath, index, col_index_named, expected_output):
    data = [[1.764052, 0.400157, 0.978738, 2.240893, 1.867558], [-0.977278, 0.950088, -0.151357, -0.103219, 0.410599]]
    df = DataFrame(data)
    if col_index_named:
        df.columns.rename('columns.name', inplace=True)
    result = df.to_html(max_cols=4, index=index)
    expected = expected_html(datapath, expected_output)
    assert result == expected