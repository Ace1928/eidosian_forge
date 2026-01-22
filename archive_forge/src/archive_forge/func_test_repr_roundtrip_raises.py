import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_repr_roundtrip_raises():
    mi = MultiIndex.from_product([list('ab'), range(3)], names=['first', 'second'])
    msg = 'Must pass both levels and codes'
    with pytest.raises(TypeError, match=msg):
        eval(repr(mi))