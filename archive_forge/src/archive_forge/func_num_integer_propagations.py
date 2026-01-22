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
@property
def num_integer_propagations(self) -> int:
    """Returns the number of integer propagations done by the solver."""
    if not self.has_response():
        raise RuntimeError('solve() has not been called.')
    return self.NumIntegerPropagations()