from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method', ['split', 'rsplit'])
def test_split_noargs(any_string_dtype, method):
    s = Series(['Wes McKinney', 'Travis  Oliphant'], dtype=any_string_dtype)
    result = getattr(s.str, method)()
    expected = ['Travis', 'Oliphant']
    assert result[1] == expected