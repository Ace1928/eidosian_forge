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
@pytest.mark.parametrize('arrow_dtype, expected_type', [[pa.binary(), bytes], [pa.binary(16), bytes], [pa.large_binary(), bytes], [pa.large_string(), str], [pa.list_(pa.int64()), list], [pa.large_list(pa.int64()), list], [pa.map_(pa.string(), pa.int64()), list], [pa.struct([('f1', pa.int8()), ('f2', pa.string())]), dict], [pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType]])
def test_arrow_dtype_type(arrow_dtype, expected_type):
    assert ArrowDtype(arrow_dtype).type == expected_type