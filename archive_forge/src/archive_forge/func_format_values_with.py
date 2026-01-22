from __future__ import annotations
from collections.abc import (
from contextlib import contextmanager
from csv import QUOTE_NONE
from decimal import Decimal
from functools import partial
from io import StringIO
import math
import re
from shutil import get_terminal_size
from typing import (
import numpy as np
from pandas._config.config import (
from pandas._libs import lib
from pandas._libs.missing import NA
from pandas._libs.tslibs import (
from pandas._libs.tslibs.nattype import NaTType
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
from pandas.core.arrays import (
from pandas.core.arrays.string_ import StringDtype
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.indexes.api import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.reshape.concat import concat
from pandas.io.common import (
from pandas.io.formats import printing
def format_values_with(float_format):
    formatter = self._value_formatter(float_format, threshold)
    na_rep = ' ' + self.na_rep if self.justify == 'left' else self.na_rep
    values = self.values
    is_complex = is_complex_dtype(values)
    if is_complex:
        values = format_complex_with_na_rep(values, formatter, na_rep)
    else:
        values = format_with_na_rep(values, formatter, na_rep)
    if self.fixed_width:
        if is_complex:
            result = _trim_zeros_complex(values, self.decimal)
        else:
            result = _trim_zeros_float(values, self.decimal)
        return np.asarray(result, dtype='object')
    return values