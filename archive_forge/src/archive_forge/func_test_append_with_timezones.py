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
def test_append_with_timezones(setup_path, gettz):
    df_est = DataFrame({'A': [Timestamp('20130102 2:00:00', tz=gettz('US/Eastern')).as_unit('ns') + timedelta(hours=1) * i for i in range(5)]})
    df_crosses_dst = DataFrame({'A': Timestamp('20130102', tz=gettz('US/Eastern')).as_unit('ns'), 'B': Timestamp('20130603', tz=gettz('US/Eastern')).as_unit('ns')}, index=range(5))
    df_mixed_tz = DataFrame({'A': Timestamp('20130102', tz=gettz('US/Eastern')).as_unit('ns'), 'B': Timestamp('20130102', tz=gettz('EET')).as_unit('ns')}, index=range(5))
    df_different_tz = DataFrame({'A': Timestamp('20130102', tz=gettz('US/Eastern')).as_unit('ns'), 'B': Timestamp('20130102', tz=gettz('CET')).as_unit('ns')}, index=range(5))
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, 'df_tz')
        store.append('df_tz', df_est, data_columns=['A'])
        result = store['df_tz']
        _compare_with_tz(result, df_est)
        tm.assert_frame_equal(result, df_est)
        expected = df_est[df_est.A >= df_est.A[3]]
        result = store.select('df_tz', where='A>=df_est.A[3]')
        _compare_with_tz(result, expected)
        _maybe_remove(store, 'df_tz')
        store.append('df_tz', df_crosses_dst)
        result = store['df_tz']
        _compare_with_tz(result, df_crosses_dst)
        tm.assert_frame_equal(result, df_crosses_dst)
        msg = 'invalid info for \\[values_block_1\\] for \\[tz\\], existing_value \\[(dateutil/.*)?US/Eastern\\] conflicts with new value \\[(dateutil/.*)?EET\\]'
        with pytest.raises(ValueError, match=msg):
            store.append('df_tz', df_mixed_tz)
        _maybe_remove(store, 'df_tz')
        store.append('df_tz', df_mixed_tz, data_columns=['A', 'B'])
        result = store['df_tz']
        _compare_with_tz(result, df_mixed_tz)
        tm.assert_frame_equal(result, df_mixed_tz)
        msg = 'invalid info for \\[B\\] for \\[tz\\], existing_value \\[(dateutil/.*)?EET\\] conflicts with new value \\[(dateutil/.*)?CET\\]'
        with pytest.raises(ValueError, match=msg):
            store.append('df_tz', df_different_tz)