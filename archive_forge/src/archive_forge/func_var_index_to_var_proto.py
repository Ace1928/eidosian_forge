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
def var_index_to_var_proto(self, var_index: int) -> cp_model_pb2.IntegerVariableProto:
    if var_index >= 0:
        return self.__model.variables[var_index]
    else:
        return self.__model.variables[-var_index - 1]