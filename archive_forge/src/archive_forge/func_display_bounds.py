import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
def display_bounds(bounds: Sequence[int]) -> str:
    """Displays a flattened list of intervals."""
    out = ''
    for i in range(0, len(bounds), 2):
        if i != 0:
            out += ', '
        if bounds[i] == bounds[i + 1]:
            out += str(bounds[i])
        else:
            out += str(bounds[i]) + '..' + str(bounds[i + 1])
    return out