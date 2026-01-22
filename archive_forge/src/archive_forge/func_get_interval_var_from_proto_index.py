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
def get_interval_var_from_proto_index(self, index: int) -> IntervalVar:
    """Returns an already created interval variable from its index."""
    if index < 0 or index >= len(self.__model.constraints):
        raise ValueError(f'get_interval_var_from_proto_index: out of bound index {index}')
    ct = self.__model.constraints[index]
    if not ct.HasField('interval'):
        raise ValueError(f'get_interval_var_from_proto_index: index {index} does not reference an' + ' interval variable')
    return IntervalVar(self.__model, index, None, None, None, None)