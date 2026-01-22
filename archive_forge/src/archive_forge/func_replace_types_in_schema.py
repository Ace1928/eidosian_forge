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
def replace_types_in_schema(schema: pa.Schema, pairs: List[Tuple[Union[Callable[[pa.DataType], bool], pa.DataType], Union[Callable[[pa.DataType], pa.DataType], pa.DataType]]], recursive: bool=True) -> pa.Schema:
    """Replace types in a schema

    :param schema: the schema
    :param pairs: a list of (is_type, convert_type) pairs
    :param recursive: whether to do recursive replacement in nested types
    :return: the new schema
    """
    fields = []
    changed = False
    for f in schema:
        new_type = f.type
        for is_type, convert_type in pairs:
            _is_type = is_type if callable(is_type) else lambda t: t == is_type
            _convert_type = convert_type if callable(convert_type) else lambda t: convert_type
            new_type = replace_type(new_type, _is_type, _convert_type, recursive=recursive)
        if f.type is new_type or f.type == new_type:
            fields.append(f)
        else:
            changed = True
            fields.append(pa.field(f.name, new_type))
    if not changed:
        return schema
    return pa.schema(fields)