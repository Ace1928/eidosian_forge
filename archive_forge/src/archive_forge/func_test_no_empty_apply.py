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
def test_no_empty_apply(mi_styler):
    mi_styler.apply(lambda s: ['a:v;'] * 2, subset=[False, False])
    mi_styler._compute()