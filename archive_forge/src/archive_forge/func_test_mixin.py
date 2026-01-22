from datetime import datetime
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.accessor import PandasDelegate
from pandas.core.base import (
def test_mixin(self):

    class T(NoNewAttributesMixin):
        pass
    t = T()
    assert not hasattr(t, '__frozen')
    t.a = 'test'
    assert t.a == 'test'
    t._freeze()
    assert '__frozen' in dir(t)
    assert getattr(t, '__frozen')
    msg = 'You cannot add any new attribute'
    with pytest.raises(AttributeError, match=msg):
        t.b = 'test'
    assert not hasattr(t, 'b')