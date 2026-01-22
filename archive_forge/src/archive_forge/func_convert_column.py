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
def convert_column(col, field):
    if field is None:
        field_nullable = True
        type_ = None
    else:
        field_nullable = field.nullable
        type_ = field.type
    try:
        result = pa.array(col, type=type_, from_pandas=True, safe=safe)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
        e.args += ('Conversion failed for column {!s} with type {!s}'.format(col.name, col.dtype),)
        raise e
    if not field_nullable and result.null_count > 0:
        raise ValueError('Field {} was non-nullable but pandas column had {} null values'.format(str(field), result.null_count))
    return result