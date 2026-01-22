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
def response_stats(self) -> str:
    """Returns some statistics on the solution found as a string."""
    return swig_helper.CpSatHelper.solver_response_stats(self._solution)