from datetime import date
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [True, False])
def test_categorical_date_roundtrip(self, box):
    v = date.today()
    obj = Index([v, v])
    assert obj.dtype == object
    if box:
        obj = obj.array
    cat = obj.astype('category')
    rtrip = cat.astype(object)
    assert rtrip.dtype == object
    assert type(rtrip[0]) is date