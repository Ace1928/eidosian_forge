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
def test_to_html_with_empty_string_label():
    data = {'c1': ['a', 'b'], 'c2': ['a', ''], 'data': [1, 2]}
    df = DataFrame(data).set_index(['c1', 'c2'])
    result = df.to_html()
    assert 'rowspan' not in result