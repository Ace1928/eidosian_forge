from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('method', ['pad', 'backfill'])
def test_fillna_method_doesnt_change_orig(self, method):
    data = np.arange(10, dtype='i8') * 24 * 3600 * 10 ** 9
    if self.array_cls is PeriodArray:
        arr = self.array_cls(data, dtype='period[D]')
    else:
        arr = self.array_cls._from_sequence(data)
    arr[4] = NaT
    fill_value = arr[3] if method == 'pad' else arr[5]
    result = arr._pad_or_backfill(method=method)
    assert result[4] == fill_value
    assert arr[4] is NaT