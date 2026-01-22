from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.mark.parametrize('text', ['String_test', True, 1, 1.0, 1j])
def test_stringify_text(text):
    valid_types = (str, int, float, bool)
    if isinstance(text, valid_types):
        result = _stringifyText(text)
        assert result == str(text)
    else:
        msg = f'only str, int, float, and bool values can be copied to the clipboard, not {type(text).__name__}'
        with pytest.raises(PyperclipException, match=msg):
            _stringifyText(text)