import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.fixture
def styler_multi(df_multi):
    return Styler(df_multi, uuid_len=0)