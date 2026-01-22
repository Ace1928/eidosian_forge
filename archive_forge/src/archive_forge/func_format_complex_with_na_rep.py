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
def format_complex_with_na_rep(values: ArrayLike, formatter: Callable, na_rep: str):
    real_values = np.real(values).ravel()
    imag_values = np.imag(values).ravel()
    real_mask, imag_mask = (isna(real_values), isna(imag_values))
    formatted_lst = []
    for val, real_val, imag_val, re_isna, im_isna in zip(values.ravel(), real_values, imag_values, real_mask, imag_mask):
        if not re_isna and (not im_isna):
            formatted_lst.append(formatter(val))
        elif not re_isna:
            formatted_lst.append(f'{formatter(real_val)}+{na_rep}j')
        elif not im_isna:
            imag_formatted = formatter(imag_val).strip()
            if imag_formatted.startswith('-'):
                formatted_lst.append(f'{na_rep}{imag_formatted}j')
            else:
                formatted_lst.append(f'{na_rep}+{imag_formatted}j')
        else:
            formatted_lst.append(f'{na_rep}+{na_rep}j')
    return np.array(formatted_lst).reshape(values.shape)