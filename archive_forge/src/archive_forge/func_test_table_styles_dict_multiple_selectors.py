import contextlib
import copy
import re
from textwrap import dedent
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.formats.style import (  # isort:skip
from pandas.io.formats.style_render import (
def test_table_styles_dict_multiple_selectors(self, df):
    result = df.style.set_table_styles({'B': [{'selector': 'th,td', 'props': [('border-left', '2px solid black')]}]})._translate(True, True)['table_styles']
    expected = [{'selector': 'th.col1', 'props': [('border-left', '2px solid black')]}, {'selector': 'td.col1', 'props': [('border-left', '2px solid black')]}]
    assert result == expected