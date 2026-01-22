from datetime import datetime
import re
from dateutil.parser import parse as du_parse
from dateutil.tz import tzlocal
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.parsing import parse_datetime_string_with_reso
from pandas.compat import (
import pandas.util._test_decorators as td
import pandas._testing as tm
from pandas._testing._hypothesis import DATETIME_NO_TZ
@pytest.mark.parametrize('input', ['2018-01-01T00:00:00.123456789', '2018-01-01T00:00:00.123456', '2018-01-01T00:00:00.123'])
def test_guess_datetime_format_f(input):
    result = parsing.guess_datetime_format(input)
    expected = '%Y-%m-%dT%H:%M:%S.%f'
    assert result == expected