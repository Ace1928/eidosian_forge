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
def test_uuid(self, df):
    styler = Styler(df, uuid='abc123')
    result = styler.to_html()
    assert 'abc123' in result
    styler = df.style
    result = styler.set_uuid('aaa')
    assert result is styler
    assert result.uuid == 'aaa'