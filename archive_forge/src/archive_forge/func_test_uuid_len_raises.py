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
@pytest.mark.parametrize('len_', [-2, 'bad', None])
def test_uuid_len_raises(self, len_):
    df = DataFrame(data=[['A']])
    msg = '``uuid_len`` must be an integer in range \\[0, 32\\].'
    with pytest.raises(TypeError, match=msg):
        Styler(df, uuid_len=len_, cell_ids=False).to_html()