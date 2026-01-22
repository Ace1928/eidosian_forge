from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.fixture
def df_ext():
    return DataFrame({'A': [0, 1, 2], 'B': [-0.61, -1.22, -2.22], 'C': ['ab', 'cd', 'de']})