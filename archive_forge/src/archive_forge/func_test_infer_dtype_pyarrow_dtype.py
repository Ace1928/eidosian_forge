from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
def test_infer_dtype_pyarrow_dtype(data, request):
    res = lib.infer_dtype(data)
    assert res != 'unknown-array'
    if data._hasna and res in ['floating', 'datetime64', 'timedelta64']:
        mark = pytest.mark.xfail(reason='in infer_dtype pd.NA is not ignored in these cases even with skipna=True in the list(data) check below')
        request.applymarker(mark)
    assert res == lib.infer_dtype(list(data), skipna=True)