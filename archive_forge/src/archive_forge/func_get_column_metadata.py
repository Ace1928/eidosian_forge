import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def get_column_metadata(column, name, arrow_type, field_name):
    """Construct the metadata for a given column

    Parameters
    ----------
    column : pandas.Series or pandas.Index
    name : str
    arrow_type : pyarrow.DataType
    field_name : str
        Equivalent to `name` when `column` is a `Series`, otherwise if `column`
        is a pandas Index then `field_name` will not be the same as `name`.
        This is the name of the field in the arrow Table's schema.

    Returns
    -------
    dict
    """
    logical_type = get_logical_type(arrow_type)
    string_dtype, extra_metadata = get_extension_dtype_info(column)
    if logical_type == 'decimal':
        extra_metadata = {'precision': arrow_type.precision, 'scale': arrow_type.scale}
        string_dtype = 'object'
    if name is not None and (not isinstance(name, str)):
        raise TypeError('Column name must be a string. Got column {} of type {}'.format(name, type(name).__name__))
    assert field_name is None or isinstance(field_name, str), str(type(field_name))
    return {'name': name, 'field_name': 'None' if field_name is None else field_name, 'pandas_type': logical_type, 'numpy_type': string_dtype, 'metadata': extra_metadata}