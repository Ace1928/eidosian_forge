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
def test_dictionary_astype_categorical():
    arrs = [pa.array(np.array(['a', 'x', 'c', 'a'])).dictionary_encode(), pa.array(np.array(['a', 'd', 'c'])).dictionary_encode()]
    ser = pd.Series(ArrowExtensionArray(pa.chunked_array(arrs)))
    result = ser.astype('category')
    categories = pd.Index(['a', 'x', 'c', 'd'], dtype=ArrowDtype(pa.string()))
    expected = pd.Series(['a', 'x', 'c', 'a', 'a', 'd', 'c'], dtype=pd.CategoricalDtype(categories=categories))
    tm.assert_series_equal(result, expected)