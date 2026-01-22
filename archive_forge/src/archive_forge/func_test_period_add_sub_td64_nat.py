from datetime import timedelta
import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas import (
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's', 'm'])
def test_period_add_sub_td64_nat(self, unit):
    per = Period('2022-06-01', 'D')
    nat = np.timedelta64('NaT', unit)
    assert per + nat is NaT
    assert nat + per is NaT
    assert per - nat is NaT
    with pytest.raises(TypeError, match='unsupported operand'):
        nat - per