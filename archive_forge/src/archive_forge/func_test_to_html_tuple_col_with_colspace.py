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
def test_to_html_tuple_col_with_colspace():
    df = DataFrame({('a', 'b'): [1], 'b': [2]})
    result = df.to_html(col_space=100)
    expected = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th style="min-width: 100px;"></th>\n      <th style="min-width: 100px;">(a, b)</th>\n      <th style="min-width: 100px;">b</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>2</td>\n    </tr>\n  </tbody>\n</table>'
    assert result == expected