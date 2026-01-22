import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_can_serialize_dates(setup_path):
    rng = [x.date() for x in bdate_range('1/1/2000', '1/30/2000')]
    frame = DataFrame(np.random.default_rng(2).standard_normal((len(rng), 4)), index=rng)
    _check_roundtrip(frame, tm.assert_frame_equal, path=setup_path)