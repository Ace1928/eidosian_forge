import json
import os
import random
import hashlib
import warnings
from typing import Union, MutableMapping, Optional, Dict, Sequence, TYPE_CHECKING, List
import pandas as pd
from toolz import curried
from typing import TypeVar
from ._importers import import_pyarrow_interchange
from .core import sanitize_dataframe, sanitize_arrow_table, DataFrameLike
from .core import sanitize_geo_interface
from .deprecation import AltairDeprecationWarning
from .plugin_registry import PluginRegistry
from typing import Protocol, TypedDict, Literal
@curried.curry
def to_values(data: DataType) -> ToValuesReturnType:
    """Replace a DataFrame by a data model with values."""
    check_data_type(data)
    if hasattr(data, '__geo_interface__'):
        if isinstance(data, pd.DataFrame):
            data = sanitize_dataframe(data)
        data_sanitized = sanitize_geo_interface(data.__geo_interface__)
        return {'values': data_sanitized}
    elif isinstance(data, pd.DataFrame):
        data = sanitize_dataframe(data)
        return {'values': data.to_dict(orient='records')}
    elif isinstance(data, dict):
        if 'values' not in data:
            raise KeyError('values expected in data dict, but not present.')
        return data
    elif isinstance(data, DataFrameLike):
        pa_table = sanitize_arrow_table(arrow_table_from_dfi_dataframe(data))
        return {'values': pa_table.to_pylist()}
    else:
        raise ValueError('Unrecognized data type: {}'.format(type(data)))