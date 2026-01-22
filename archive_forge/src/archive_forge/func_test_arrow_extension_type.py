import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_arrow_extension_type():
    from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
    p1 = ArrowPeriodType('D')
    p2 = ArrowPeriodType('D')
    p3 = ArrowPeriodType('M')
    assert p1.freq == 'D'
    assert p1 == p2
    assert p1 != p3
    assert hash(p1) == hash(p2)
    assert hash(p1) != hash(p3)