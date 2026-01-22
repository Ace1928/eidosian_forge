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
@td.skip_if_not_us_locale
@pytest.mark.parametrize('string,fmt', [('20111230', '%Y%m%d'), ('201112300000', '%Y%m%d%H%M'), ('20111230000000', '%Y%m%d%H%M%S'), ('20111230T00', '%Y%m%dT%H'), ('20111230T0000', '%Y%m%dT%H%M'), ('20111230T000000', '%Y%m%dT%H%M%S'), ('2011-12-30', '%Y-%m-%d'), ('2011', '%Y'), ('2011-01', '%Y-%m'), ('30-12-2011', '%d-%m-%Y'), ('2011-12-30 00:00:00', '%Y-%m-%d %H:%M:%S'), ('2011-12-30T00:00:00', '%Y-%m-%dT%H:%M:%S'), ('2011-12-30T00:00:00UTC', '%Y-%m-%dT%H:%M:%S%Z'), ('2011-12-30T00:00:00Z', '%Y-%m-%dT%H:%M:%S%z'), ('2011-12-30T00:00:00+9', '%Y-%m-%dT%H:%M:%S%z'), ('2011-12-30T00:00:00+09', '%Y-%m-%dT%H:%M:%S%z'), ('2011-12-30T00:00:00+090', None), ('2011-12-30T00:00:00+0900', '%Y-%m-%dT%H:%M:%S%z'), ('2011-12-30T00:00:00-0900', '%Y-%m-%dT%H:%M:%S%z'), ('2011-12-30T00:00:00+09:00', '%Y-%m-%dT%H:%M:%S%z'), ('2011-12-30T00:00:00+09:000', None), ('2011-12-30T00:00:00+9:0', '%Y-%m-%dT%H:%M:%S%z'), ('2011-12-30T00:00:00+09:', None), ('2011-12-30T00:00:00.000000UTC', '%Y-%m-%dT%H:%M:%S.%f%Z'), ('2011-12-30T00:00:00.000000Z', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2011-12-30T00:00:00.000000+9', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2011-12-30T00:00:00.000000+09', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2011-12-30T00:00:00.000000+090', None), ('2011-12-30T00:00:00.000000+0900', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2011-12-30T00:00:00.000000-0900', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2011-12-30T00:00:00.000000+09:00', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2011-12-30T00:00:00.000000+09:000', None), ('2011-12-30T00:00:00.000000+9:0', '%Y-%m-%dT%H:%M:%S.%f%z'), ('2011-12-30T00:00:00.000000+09:', None), ('2011-12-30 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f'), ('Tue 24 Aug 2021 01:30:48', '%a %d %b %Y %H:%M:%S'), ('Tuesday 24 Aug 2021 01:30:48', '%A %d %b %Y %H:%M:%S'), ('Tue 24 Aug 2021 01:30:48 AM', '%a %d %b %Y %I:%M:%S %p'), ('Tuesday 24 Aug 2021 01:30:48 AM', '%A %d %b %Y %I:%M:%S %p'), ('27.03.2003 14:55:00.000', '%d.%m.%Y %H:%M:%S.%f')])
def test_guess_datetime_format_with_parseable_formats(string, fmt):
    with tm.maybe_produces_warning(UserWarning, fmt is not None and re.search('%d.*%m', fmt)):
        result = parsing.guess_datetime_format(string)
    assert result == fmt