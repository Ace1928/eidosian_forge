from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('data, pat', [(['bd asdf jfg', 'kjasdflqw asdfnfk'], None), (['bd asdf jfg', 'kjasdflqw asdfnfk'], 'asdf'), (['bd_asdf_jfg', 'kjasdflqw_asdfnfk'], '_')])
@pytest.mark.parametrize('n', [-1, 0])
def test_split_maxsplit(data, pat, any_string_dtype, n):
    s = Series(data, dtype=any_string_dtype)
    result = s.str.split(pat=pat, n=n)
    xp = s.str.split(pat=pat)
    tm.assert_series_equal(result, xp)