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
@pytest.mark.skipif(is_platform_windows() or ISMUSL, reason='TZ setting incorrect on Windows and MUSL Linux')
def test_parsing_tzlocal_deprecated():
    msg = "Parsing 'EST' as tzlocal.*Pass the 'tz' keyword or call tz_localize after construction instead"
    dtstr = 'Jan 15 2004 03:00 EST'
    with tm.set_timezone('US/Eastern'):
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res, _ = parse_datetime_string_with_reso(dtstr)
        assert isinstance(res.tzinfo, tzlocal)
        with tm.assert_produces_warning(FutureWarning, match=msg):
            res = parsing.py_parse_datetime_string(dtstr)
        assert isinstance(res.tzinfo, tzlocal)