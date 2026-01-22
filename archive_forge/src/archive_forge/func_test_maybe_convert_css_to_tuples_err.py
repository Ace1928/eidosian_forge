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
def test_maybe_convert_css_to_tuples_err(self):
    msg = 'Styles supplied as string must follow CSS rule formats'
    with pytest.raises(ValueError, match=msg):
        maybe_convert_css_to_tuples('err')