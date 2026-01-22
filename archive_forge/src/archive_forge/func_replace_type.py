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
def replace_type(current_type: pa.DataType, is_type: Callable[[pa.DataType], bool], convert_type: Callable[[pa.DataType], pa.DataType], recursive: bool=True) -> pa.DataType:
    """Replace ``current_type`` or if it is nested, replace in the nested
    types

    :param current_type: the current type
    :param is_type: the function to check if the type is the type to replace
    :param convert_type: the function to convert the type
    :param recursive: whether to do recursive replacement in nested types
    :return: the new type
    """
    if not pa.types.is_nested(current_type) and is_type(current_type):
        return convert_type(current_type)
    if recursive:
        if pa.types.is_struct(current_type):
            old_fields = list(current_type)
            fields = [_replace_field(f, is_type, convert_type, recursive=recursive) for f in old_fields]
            if all((a is b for a, b in zip(fields, old_fields))):
                return current_type
            return pa.struct(fields)
        if pa.types.is_list(current_type) or pa.types.is_large_list(current_type):
            old_f = current_type.value_field
            f = _replace_field(old_f, is_type, convert_type, recursive=recursive)
            if f is old_f:
                res = current_type
            elif pa.types.is_large_list(current_type):
                res = pa.large_list(f)
            else:
                res = pa.list_(f)
            if is_type(res):
                return convert_type(res)
            return res
        if pa.types.is_map(current_type):
            old_k, old_v = (current_type.key_field, current_type.item_field)
            k, v = (_replace_field(old_k, is_type, convert_type, recursive=recursive), _replace_field(old_v, is_type, convert_type, recursive=recursive))
            if k is old_k and v is old_v:
                return current_type
            return pa.map_(k, v)
    return current_type