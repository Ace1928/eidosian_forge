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
def test_to_html_float_format_object_col(datapath):
    df = DataFrame(data={'x': [1000.0, 'test']})
    result = df.to_html(float_format=lambda x: f'{x:,.0f}')
    expected = expected_html(datapath, 'gh40024_expected_output')
    assert result == expected