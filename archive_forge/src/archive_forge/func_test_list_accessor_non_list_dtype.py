import re
import pytest
from pandas import (
import pandas._testing as tm
from pandas.compat import pa_version_under11p0
def test_list_accessor_non_list_dtype():
    ser = Series([1, 2, 4], dtype=ArrowDtype(pa.int64()))
    with pytest.raises(AttributeError, match=re.escape("Can only use the '.list' accessor with 'list[pyarrow]' dtype, not int64[pyarrow].")):
        ser.list[1:None:0]