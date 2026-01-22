from __future__ import annotations
from datetime import datetime
import numpy as np
import pytest
from pandas import Timestamp
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
def test_misspecified(self):
    with pytest.raises(ValueError, match='Month must go from 1 to 12'):
        YearEnd(month=13)