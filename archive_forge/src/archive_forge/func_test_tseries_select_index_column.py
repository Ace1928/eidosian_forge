from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_tseries_select_index_column(setup_path):
    rng = date_range('1/1/2000', '1/30/2000')
    frame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng)
    with ensure_clean_store(setup_path) as store:
        store.append('frame', frame)
        result = store.select_column('frame', 'index')
        assert rng.tz == DatetimeIndex(result.values).tz
    rng = date_range('1/1/2000', '1/30/2000', tz='UTC')
    frame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng)
    with ensure_clean_store(setup_path) as store:
        store.append('frame', frame)
        result = store.select_column('frame', 'index')
        assert rng.tz == result.dt.tz
    rng = date_range('1/1/2000', '1/30/2000', tz='US/Eastern')
    frame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng)
    with ensure_clean_store(setup_path) as store:
        store.append('frame', frame)
        result = store.select_column('frame', 'index')
        assert rng.tz == result.dt.tz