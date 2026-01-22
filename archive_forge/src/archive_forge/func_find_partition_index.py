import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def find_partition_index(table: Union['pyarrow.Table', 'pandas.DataFrame'], desired: List[Any], sort_key: 'SortKey') -> int:
    columns = sort_key.get_columns()
    descending = sort_key.get_descending()
    left, right = (0, len(table))
    for i in range(len(desired)):
        if left == right:
            return right
        col_name = columns[i]
        col_vals = table[col_name].to_numpy()[left:right]
        desired_val = desired[i]
        prevleft = left
        if descending is True:
            left = prevleft + (len(col_vals) - np.searchsorted(col_vals, desired_val, side='right', sorter=np.arange(len(col_vals) - 1, -1, -1)))
            right = prevleft + (len(col_vals) - np.searchsorted(col_vals, desired_val, side='left', sorter=np.arange(len(col_vals) - 1, -1, -1)))
        else:
            left = prevleft + np.searchsorted(col_vals, desired_val, side='left')
            right = prevleft + np.searchsorted(col_vals, desired_val, side='right')
    return right if descending is True else left