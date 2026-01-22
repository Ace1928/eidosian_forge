import pytest
from pandas import (
import pandas._testing as tm
def test_astype_invalid_nas_to_tdt64_raises():
    idx = Index([NaT.asm8] * 2, dtype=object)
    msg = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
    with pytest.raises(TypeError, match=msg):
        idx.astype('m8[ns]')