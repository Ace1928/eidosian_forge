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
@pytest.mark.parametrize('len_', [1, 5, 32, 33, 100])
def test_uuid_len(self, len_):
    df = DataFrame(data=[['A']])
    s = Styler(df, uuid_len=len_, cell_ids=False).to_html()
    strt = s.find('id="T_')
    end = s[strt + 6:].find('"')
    if len_ > 32:
        assert end == 32
    else:
        assert end == len_