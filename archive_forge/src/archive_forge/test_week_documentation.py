from __future__ import annotations
from datetime import (
import pytest
from pandas._libs.tslibs import Timestamp
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (

Tests for the following offsets:
- Week
- WeekOfMonth
- LastWeekOfMonth
