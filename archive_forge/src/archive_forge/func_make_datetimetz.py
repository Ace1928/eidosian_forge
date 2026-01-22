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
def make_datetimetz(unit, tz):
    if _pandas_api.is_v1():
        unit = 'ns'
    tz = pa.lib.string_to_tzinfo(tz)
    return _pandas_api.datetimetz_type(unit, tz=tz)