from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
@pytest.mark.parametrize('unit', ['Y', 'y', 'M'])
def test_unit_m_y_raises(self, unit):
    msg = "Units 'M', 'Y', and 'y' are no longer supported"
    depr_msg = "The 'unit' keyword in TimedeltaIndex construction is deprecated"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            TimedeltaIndex([1, 3, 7], unit)