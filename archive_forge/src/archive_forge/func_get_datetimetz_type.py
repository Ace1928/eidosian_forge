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
def get_datetimetz_type(values, dtype, type_):
    if values.dtype.type != np.datetime64:
        return (values, type_)
    if _pandas_api.is_datetimetz(dtype) and type_ is None:
        tz = dtype.tz
        unit = dtype.unit
        type_ = pa.timestamp(unit, tz)
    elif type_ is None:
        type_ = pa.from_numpy_dtype(values.dtype)
    return (values, type_)