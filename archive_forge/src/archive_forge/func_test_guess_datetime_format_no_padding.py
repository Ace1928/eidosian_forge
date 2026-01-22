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
@pytest.mark.parametrize('string,fmt,dayfirst,warning', [('2011-1-1', '%Y-%m-%d', False, None), ('2011-1-1', '%Y-%d-%m', True, None), ('1/1/2011', '%m/%d/%Y', False, None), ('1/1/2011', '%d/%m/%Y', True, None), ('30-1-2011', '%d-%m-%Y', False, UserWarning), ('30-1-2011', '%d-%m-%Y', True, None), ('2011-1-1 0:0:0', '%Y-%m-%d %H:%M:%S', False, None), ('2011-1-1 0:0:0', '%Y-%d-%m %H:%M:%S', True, None), ('2011-1-3T00:00:0', '%Y-%m-%dT%H:%M:%S', False, None), ('2011-1-3T00:00:0', '%Y-%d-%mT%H:%M:%S', True, None), ('2011-1-1 00:00:00', '%Y-%m-%d %H:%M:%S', False, None), ('2011-1-1 00:00:00', '%Y-%d-%m %H:%M:%S', True, None)])
def test_guess_datetime_format_no_padding(string, fmt, dayfirst, warning):
    msg = f'Parsing dates in {fmt} format when dayfirst=False \\(the default\\) was specified. Pass `dayfirst=True` or specify a format to silence this warning.'
    with tm.assert_produces_warning(warning, match=msg):
        result = parsing.guess_datetime_format(string, dayfirst=dayfirst)
    assert result == fmt