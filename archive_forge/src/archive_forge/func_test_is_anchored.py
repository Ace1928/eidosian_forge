from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_is_anchored(self):
    msg = 'BQuarterEnd.is_anchored is deprecated '
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert BQuarterEnd(startingMonth=1).is_anchored()
        assert BQuarterEnd().is_anchored()
        assert not BQuarterEnd(2, startingMonth=1).is_anchored()