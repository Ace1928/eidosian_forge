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
def get_logical_type_from_numpy(pandas_collection):
    try:
        return _numpy_logical_type_map[pandas_collection.dtype.type]
    except KeyError:
        if hasattr(pandas_collection.dtype, 'tz'):
            return 'datetimetz'
        if str(pandas_collection.dtype) == 'datetime64[ns]':
            return 'datetime64[ns]'
        result = _pandas_api.infer_dtype(pandas_collection)
        if result == 'string':
            return 'unicode'
        return result