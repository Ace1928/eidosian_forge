from __future__ import annotations
import functools
import operator
import re
import textwrap
from typing import (
import unicodedata
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas.compat import (
from pandas.util._decorators import doc
from pandas.util._validators import validate_fillna_kwargs
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import isna
from pandas.core import (
from pandas.core.algorithms import map_array
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas.core.arrays.base import (
from pandas.core.arrays.masked import BaseMaskedArray
from pandas.core.arrays.string_ import StringDtype
import pandas.core.common as com
from pandas.core.indexers import (
from pandas.core.strings.base import BaseStringArrayMethods
from pandas.io._util import _arrow_dtype_mapping
from pandas.tseries.frequencies import to_offset
@doc(ExtensionArray.factorize)
def factorize(self, use_na_sentinel: bool=True) -> tuple[np.ndarray, ExtensionArray]:
    null_encoding = 'mask' if use_na_sentinel else 'encode'
    data = self._pa_array
    pa_type = data.type
    if pa_version_under11p0 and pa.types.is_duration(pa_type):
        data = data.cast(pa.int64())
    if pa.types.is_dictionary(data.type):
        encoded = data
    else:
        encoded = data.dictionary_encode(null_encoding=null_encoding)
    if encoded.length() == 0:
        indices = np.array([], dtype=np.intp)
        uniques = type(self)(pa.chunked_array([], type=encoded.type.value_type))
    else:
        combined = encoded.combine_chunks()
        pa_indices = combined.indices
        if pa_indices.null_count > 0:
            pa_indices = pc.fill_null(pa_indices, -1)
        indices = pa_indices.to_numpy(zero_copy_only=False, writable=True).astype(np.intp, copy=False)
        uniques = type(self)(combined.dictionary)
    if pa_version_under11p0 and pa.types.is_duration(pa_type):
        uniques = cast(ArrowExtensionArray, uniques.astype(self.dtype))
    return (indices, uniques)