from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('repl', [None, 3, {'a': 'b'}])
@pytest.mark.parametrize('data', [['a', 'b', None], ['a', 'b', 'c', 'ad']])
def test_replace_wrong_repl_type_raises(any_string_dtype, index_or_series, repl, data):
    msg = 'repl must be a string or callable'
    obj = index_or_series(data, dtype=any_string_dtype)
    with pytest.raises(TypeError, match=msg):
        obj.str.replace('a', repl)