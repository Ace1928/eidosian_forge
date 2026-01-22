import calendar
from collections import deque
from datetime import (
from decimal import Decimal
import locale
from dateutil.parser import parse
from dateutil.tz.tz import tzoffset
import numpy as np
import pytest
import pytz
from pandas._libs import tslib
from pandas._libs.tslibs import (
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_datetime64_ns_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.core.tools import datetimes as tools
from pandas.core.tools.datetimes import start_caching_at
@pytest.mark.parametrize('msg, s, _format', [["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 50', '%Y %V'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51', '%G %V'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Monday', '%G %A'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 Mon', '%G %a'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %w'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 6', '%G %u'], ["ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.", '2051', '%G'], ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 51 6 256', '%G %V %u %j'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sunday', '%Y %V %A'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 Sun', '%Y %V %a'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %w'], ["ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.", '1999 51 1', '%Y %V %u'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20', '%V'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sunday', '%V %A'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 Sun', '%V %a'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %w'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '1999 51 1', '%V %u'], ["Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.", '1999 50', '%G %j'], ["ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.", '20 Monday', '%V %A']])
@pytest.mark.parametrize('errors', ['raise', 'coerce', 'ignore'])
def test_error_iso_week_year(self, msg, s, _format, errors):
    if locale.getlocale() != ('zh_CN', 'UTF-8') and locale.getlocale() != ('it_IT', 'UTF-8'):
        with pytest.raises(ValueError, match=msg):
            to_datetime(s, format=_format, errors=errors)