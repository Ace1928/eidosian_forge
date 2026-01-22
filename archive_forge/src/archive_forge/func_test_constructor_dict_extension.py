from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_dict_extension(self, ea_scalar_and_dtype, request):
    ea_scalar, ea_dtype = ea_scalar_and_dtype
    if isinstance(ea_scalar, Timestamp):
        mark = pytest.mark.xfail(reason='Construction from dict goes through maybe_convert_objects which casts to nano')
        request.applymarker(mark)
    d = {'a': ea_scalar}
    result = Series(d, index=['a'])
    expected = Series(ea_scalar, index=['a'], dtype=ea_dtype)
    assert result.dtype == ea_dtype
    tm.assert_series_equal(result, expected)