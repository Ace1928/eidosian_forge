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
def rebuild_constant_map(self):
    """Internal method used during model cloning."""
    for i, var in enumerate(self.__model.variables):
        if len(var.domain) == 2 and var.domain[0] == var.domain[1]:
            self.__constant_map[var.domain[0]] = i