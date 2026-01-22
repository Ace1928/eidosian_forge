from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
def testRollback1(self):
    assert CDay(10).rollback(datetime(2007, 12, 31)) == datetime(2007, 12, 31)