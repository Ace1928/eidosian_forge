import json
import pickle
from datetime import date, datetime
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow.compute import CastOptions, binary_join_element_wise
from pyarrow.json import read_json, ParseOptions as JsonParseOptions
from triad.constants import TRIAD_VAR_QUOTE
from .convert import as_type
from .iter import EmptyAwareIterable, Slicer
from .json import loads_no_dup
from .schema import move_to_unquoted, quote_name, unquote_name
from .assertion import assert_or_throw
def to_single_pandas_dtype(pa_type: pa.DataType, use_extension_types: bool=False, use_arrow_dtype: bool=False) -> np.dtype:
    """convert a pyarrow data type to a pandas datatype.
    Currently, struct type is not supported

    :param pa_type: the pyarrow data type
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :return: the pandas data type

    .. note::

        * If ``use_extension_types`` is False and ``use_arrow_dtype`` is True,
            it converts the type to ``ArrowDType``
        * If both are true, it converts the type to the numpy backend nullable
            dtypes if possible, otherwise, it converts to ``ArrowDType``
    """
    use_arrow_dtype = use_arrow_dtype and hasattr(pd, 'ArrowDtype')
    if pa.types.is_nested(pa_type) and (not use_arrow_dtype):
        return np.dtype(object)
    tp = to_pandas_types_mapper(pa_type, use_extension_types=use_extension_types, use_arrow_dtype=use_arrow_dtype)
    if tp is not None:
        return tp
    if pa.types.is_string(pa_type) and (not use_extension_types) and (not use_arrow_dtype):
        return np.dtype(str)
    return pa_type.to_pandas_dtype()