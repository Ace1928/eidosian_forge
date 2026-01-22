from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
@pytest.mark.parametrize('gettz', [gettz_dateutil, gettz_pytz])
def test_append_with_timezones_as_index(setup_path, gettz):
    dti = date_range('2000-1-1', periods=3, freq='h', tz=gettz('US/Eastern'))
    dti = dti._with_freq(None)
    df = DataFrame({'A': Series(range(3), index=dti)})
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, 'df')
        store.put('df', df)
        result = store.select('df')
        tm.assert_frame_equal(result, df)
        _maybe_remove(store, 'df')
        store.append('df', df)
        result = store.select('df')
        tm.assert_frame_equal(result, df)