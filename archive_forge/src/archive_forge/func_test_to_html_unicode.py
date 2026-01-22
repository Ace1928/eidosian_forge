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
@pytest.mark.parametrize('df,expected', [(DataFrame({'σ': np.arange(10.0)}), 'unicode_1'), (DataFrame({'A': ['σ']}), 'unicode_2')])
def test_to_html_unicode(df, expected, datapath):
    expected = expected_html(datapath, expected)
    result = df.to_html()
    assert result == expected