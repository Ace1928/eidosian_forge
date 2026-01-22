from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
def test_fixed_offset_tz(setup_path):
    rng = date_range('1/1/2000 00:00:00-07:00', '1/30/2000 00:00:00-07:00')
    frame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng)
    with ensure_clean_store(setup_path) as store:
        store['frame'] = frame
        recons = store['frame']
        tm.assert_index_equal(recons.index, rng)
        assert rng.tz == recons.index.tz