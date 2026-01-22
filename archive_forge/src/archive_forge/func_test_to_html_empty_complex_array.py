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
def test_to_html_empty_complex_array():
    df = DataFrame({'x': np.array([], dtype='complex')})
    result = df.to_html(col_space=100)
    expected = '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th style="min-width: 100px;"></th>\n      <th style="min-width: 100px;">x</th>\n    </tr>\n  </thead>\n  <tbody>\n  </tbody>\n</table>'
    assert result == expected