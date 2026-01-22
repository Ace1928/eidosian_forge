from __future__ import annotations
import decimal
import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.decimal.array import (
@pytest.mark.parametrize('frame', [True, False])
def test_astype_dispatches(frame):
    data = pd.Series(DecimalArray([decimal.Decimal(2)]), name='a')
    ctx = decimal.Context()
    ctx.prec = 5
    if frame:
        data = data.to_frame()
    result = data.astype(DecimalDtype(ctx))
    if frame:
        result = result['a']
    assert result.dtype.context.prec == ctx.prec