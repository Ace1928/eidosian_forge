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
def test_table_styles(self, df):
    style = [{'selector': 'th', 'props': [('foo', 'bar')]}]
    styler = Styler(df, table_styles=style)
    result = ' '.join(styler.to_html().split())
    assert 'th { foo: bar; }' in result
    styler = df.style
    result = styler.set_table_styles(style)
    assert styler is result
    assert styler.table_styles == style
    style = [{'selector': 'th', 'props': 'foo:bar;'}]
    styler = df.style.set_table_styles(style)
    result = ' '.join(styler.to_html().split())
    assert 'th { foo: bar; }' in result