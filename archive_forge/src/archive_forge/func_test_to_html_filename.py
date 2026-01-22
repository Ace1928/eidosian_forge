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
def test_to_html_filename(biggie_df_fixture, tmpdir):
    df = biggie_df_fixture
    expected = df.to_html()
    path = tmpdir.join('test.html')
    df.to_html(path)
    result = path.read()
    assert result == expected