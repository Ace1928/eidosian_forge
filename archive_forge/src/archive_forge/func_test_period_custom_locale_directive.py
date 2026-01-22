from contextlib import nullcontext
from datetime import (
import locale
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('locale_str', [pytest.param(None, id=str(locale.getlocale())), 'it_IT.utf8', 'it_IT', 'zh_CN.utf8', 'zh_CN'])
def test_period_custom_locale_directive(self, locale_str):
    if locale_str is not None and (not tm.can_set_locale(locale_str, locale.LC_ALL)):
        pytest.skip(f"Skipping as locale '{locale_str}' cannot be set on host.")
    with tm.set_locale(locale_str, locale.LC_ALL) if locale_str else nullcontext():
        am_local, pm_local = get_local_am_pm()
        per = pd.Period('2018-03-11 13:00', freq='h')
        assert per.strftime('%p') == pm_local
        per = pd.period_range('2003-01-01 01:00:00', periods=2, freq='12h')
        msg = 'PeriodIndex.format is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            formatted = per.format(date_format='%y %I:%M:%S%p')
        assert formatted[0] == f'03 01:00:00{am_local}'
        assert formatted[1] == f'03 01:00:00{pm_local}'