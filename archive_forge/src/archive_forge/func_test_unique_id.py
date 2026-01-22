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
def test_unique_id(self):
    df = DataFrame({'a': [1, 3, 5, 6], 'b': [2, 4, 12, 21]})
    result = df.style.to_html(uuid='test')
    assert 'test' in result
    ids = re.findall('id="(.*?)"', result)
    assert np.unique(ids).size == len(ids)