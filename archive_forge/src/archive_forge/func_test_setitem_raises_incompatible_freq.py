import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_setitem_raises_incompatible_freq():
    arr = PeriodArray(np.arange(3), dtype='period[D]')
    with pytest.raises(IncompatibleFrequency, match='freq'):
        arr[0] = pd.Period('2000', freq='Y')
    other = PeriodArray._from_sequence(['2000', '2001'], dtype='period[Y]')
    with pytest.raises(IncompatibleFrequency, match='freq'):
        arr[[0, 1]] = other