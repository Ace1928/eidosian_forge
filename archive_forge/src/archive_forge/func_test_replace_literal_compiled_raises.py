from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_literal_compiled_raises(any_string_dtype):
    ser = Series([], dtype=any_string_dtype)
    pat = re.compile('[a-z][A-Z]{2}')
    msg = 'Cannot use a compiled regex as replacement pattern with regex=False'
    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, '', regex=False)