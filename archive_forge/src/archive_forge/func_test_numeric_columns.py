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
def test_numeric_columns(self):
    df = DataFrame({0: [1, 2, 3]})
    df.style._translate(True, True)