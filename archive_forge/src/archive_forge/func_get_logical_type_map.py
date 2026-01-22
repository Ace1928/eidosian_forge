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
def get_logical_type_map():
    global _logical_type_map
    if not _logical_type_map:
        _logical_type_map.update({pa.lib.Type_NA: 'empty', pa.lib.Type_BOOL: 'bool', pa.lib.Type_INT8: 'int8', pa.lib.Type_INT16: 'int16', pa.lib.Type_INT32: 'int32', pa.lib.Type_INT64: 'int64', pa.lib.Type_UINT8: 'uint8', pa.lib.Type_UINT16: 'uint16', pa.lib.Type_UINT32: 'uint32', pa.lib.Type_UINT64: 'uint64', pa.lib.Type_HALF_FLOAT: 'float16', pa.lib.Type_FLOAT: 'float32', pa.lib.Type_DOUBLE: 'float64', pa.lib.Type_DATE32: 'date', pa.lib.Type_DATE64: 'date', pa.lib.Type_TIME32: 'time', pa.lib.Type_TIME64: 'time', pa.lib.Type_BINARY: 'bytes', pa.lib.Type_FIXED_SIZE_BINARY: 'bytes', pa.lib.Type_STRING: 'unicode'})
    return _logical_type_map