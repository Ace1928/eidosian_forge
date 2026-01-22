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
def get_alter_func(from_schema: pa.Schema, to_schema: pa.Schema, safe: bool) -> Callable[[pa.Table], pa.Table]:
    """Generate the alteration function based on ``from_schema`` and
    ``to_schema``. This function can be applied to arrow tables with
    ``from_schema``, the outout will be in ``to_schema``'s order and types

    :param from_schema: the source schema
    :param to_schema: the destination schema
    :param safe: whether to check for conversion errors such as overflow
    :return: a function that can be applied to arrow tables with
        ``from_schema``, the outout will be in ``to_schema``'s order
        and types
    """
    params: List[Tuple[pa.Field, int, bool]] = []
    same = True
    for i, f in enumerate(to_schema):
        j = from_schema.get_field_index(f.name)
        if j < 0:
            raise KeyError(f'{f.name} is not in {from_schema}')
        other = from_schema.field(j)
        pos_same, type_same = (i == j, f.type == other.type)
        same &= pos_same & type_same
        params.append((f, j, type_same))

    def _alter(df: pa.Table, params: List[Tuple[pa.Field, int, bool]]) -> pa.Table:
        cols: List[pa.ChunkedArray] = []
        names: List[str] = []
        for field, pos, same in params:
            names.append(field.name)
            col = df.column(pos)
            if not same:
                col = col.cast(field.type, safe=safe)
            cols.append(col)
        return pa.Table.from_arrays(cols, names=names)
    if same:
        return lambda x: x
    return partial(_alter, params=params)