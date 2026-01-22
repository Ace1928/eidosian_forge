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
@pytest.mark.filterwarnings('ignore:elementwise comparison failed:DeprecationWarning')
@pytest.mark.parametrize('func', [Series, DataFrame, Index, pd.array])
def test_constructor_mismatched_null_nullable_dtype(self, func, any_numeric_ea_dtype):
    msg = '|'.join(['cannot safely cast non-equivalent object', 'int\\(\\) argument must be a string, a bytes-like object or a (real )?number', "Cannot cast array data from dtype\\('O'\\) to dtype\\('float64'\\) according to the rule 'safe'", 'object cannot be converted to a FloatingDtype', "'values' contains non-numeric NA"])
    for null in tm.NP_NAT_OBJECTS + [NaT]:
        with pytest.raises(TypeError, match=msg):
            func([null, 1.0, 3.0], dtype=any_numeric_ea_dtype)