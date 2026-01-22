from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def np_can_hold_element(dtype: np.dtype, element: Any) -> Any:
    """
    Raise if we cannot losslessly set this element into an ndarray with this dtype.

    Specifically about places where we disagree with numpy.  i.e. there are
    cases where numpy will raise in doing the setitem that we do not check
    for here, e.g. setting str "X" into a numeric ndarray.

    Returns
    -------
    Any
        The element, potentially cast to the dtype.

    Raises
    ------
    ValueError : If we cannot losslessly store this element with this dtype.
    """
    if dtype == _dtype_obj:
        return element
    tipo = _maybe_infer_dtype_type(element)
    if dtype.kind in 'iu':
        if isinstance(element, range):
            if _dtype_can_hold_range(element, dtype):
                return element
            raise LossySetitemError
        if is_integer(element) or (is_float(element) and element.is_integer()):
            info = np.iinfo(dtype)
            if info.min <= element <= info.max:
                return dtype.type(element)
            raise LossySetitemError
        if tipo is not None:
            if tipo.kind not in 'iu':
                if isinstance(element, np.ndarray) and element.dtype.kind == 'f':
                    with np.errstate(invalid='ignore'):
                        casted = element.astype(dtype)
                    comp = casted == element
                    if comp.all():
                        return casted
                    raise LossySetitemError
                elif isinstance(element, ABCExtensionArray) and isinstance(element.dtype, CategoricalDtype):
                    try:
                        casted = element.astype(dtype)
                    except (ValueError, TypeError):
                        raise LossySetitemError
                    comp = casted == element
                    if not comp.all():
                        raise LossySetitemError
                    return casted
                raise LossySetitemError
            if dtype.kind == 'u' and isinstance(element, np.ndarray) and (element.dtype.kind == 'i'):
                casted = element.astype(dtype)
                if (casted == element).all():
                    return casted
                raise LossySetitemError
            if dtype.itemsize < tipo.itemsize:
                raise LossySetitemError
            if not isinstance(tipo, np.dtype):
                arr = element._values if isinstance(element, ABCSeries) else element
                if arr._hasna:
                    raise LossySetitemError
                return element
            return element
        raise LossySetitemError
    if dtype.kind == 'f':
        if lib.is_integer(element) or lib.is_float(element):
            casted = dtype.type(element)
            if np.isnan(casted) or casted == element:
                return casted
            raise LossySetitemError
        if tipo is not None:
            if tipo.kind not in 'iuf':
                raise LossySetitemError
            if not isinstance(tipo, np.dtype):
                if element._hasna:
                    raise LossySetitemError
                return element
            elif tipo.itemsize > dtype.itemsize or tipo.kind != dtype.kind:
                if isinstance(element, np.ndarray):
                    casted = element.astype(dtype)
                    if np.array_equal(casted, element, equal_nan=True):
                        return casted
                    raise LossySetitemError
            return element
        raise LossySetitemError
    if dtype.kind == 'c':
        if lib.is_integer(element) or lib.is_complex(element) or lib.is_float(element):
            if np.isnan(element):
                return dtype.type(element)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                casted = dtype.type(element)
            if casted == element:
                return casted
            raise LossySetitemError
        if tipo is not None:
            if tipo.kind in 'iufc':
                return element
            raise LossySetitemError
        raise LossySetitemError
    if dtype.kind == 'b':
        if tipo is not None:
            if tipo.kind == 'b':
                if not isinstance(tipo, np.dtype):
                    if element._hasna:
                        raise LossySetitemError
                return element
            raise LossySetitemError
        if lib.is_bool(element):
            return element
        raise LossySetitemError
    if dtype.kind == 'S':
        if tipo is not None:
            if tipo.kind == 'S' and tipo.itemsize <= dtype.itemsize:
                return element
            raise LossySetitemError
        if isinstance(element, bytes) and len(element) <= dtype.itemsize:
            return element
        raise LossySetitemError
    if dtype.kind == 'V':
        raise LossySetitemError
    raise NotImplementedError(dtype)