from collections.abc import Mapping, MutableMapping
from copy import deepcopy
import json
import itertools
import re
import sys
import traceback
import warnings
from typing import (
from types import ModuleType
import jsonschema
import pandas as pd
import numpy as np
from pandas.api.types import infer_dtype
from altair.utils.schemapi import SchemaBase
from altair.utils._dfi_types import Column, DtypeKind, DataFrame as DfiDataFrame
from typing import Literal, Protocol, TYPE_CHECKING, runtime_checkable
def infer_vegalite_type_for_dfi_column(column: Union[Column, 'PandasColumn']) -> Union[InferredVegaLiteType, Tuple[InferredVegaLiteType, list]]:
    from pyarrow.interchange.from_dataframe import column_to_array
    try:
        kind = column.dtype[0]
    except NotImplementedError as e:
        if 'datetime64' in e.args[0] or 'timestamp' in e.args[0]:
            return 'temporal'
        raise e
    if kind == DtypeKind.CATEGORICAL and column.describe_categorical['is_ordered'] and (column.describe_categorical['categories'] is not None):
        categories_column = column.describe_categorical['categories']
        categories_array = column_to_array(categories_column)
        return ('ordinal', categories_array.to_pylist())
    if kind in (DtypeKind.STRING, DtypeKind.CATEGORICAL, DtypeKind.BOOL):
        return 'nominal'
    elif kind in (DtypeKind.INT, DtypeKind.UINT, DtypeKind.FLOAT):
        return 'quantitative'
    elif kind == DtypeKind.DATETIME:
        return 'temporal'
    else:
        raise ValueError(f'Unexpected DtypeKind: {kind}')