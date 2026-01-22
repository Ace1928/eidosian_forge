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
@pytest.mark.parametrize('dashed', ['-2Q1992', '2-Q1992', '4-4Q1992'])
def test_parse_time_quarter_with_dash_error(dashed):
    msg = f'Unknown datetime string format, unable to parse: {dashed}'
    with pytest.raises(parsing.DateParseError, match=msg):
        parse_datetime_string_with_reso(dashed)