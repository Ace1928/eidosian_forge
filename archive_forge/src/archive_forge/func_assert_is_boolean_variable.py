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
def assert_is_boolean_variable(self, x: LiteralT) -> None:
    if isinstance(x, IntVar):
        var = self.__model.variables[x.index]
        if len(var.domain) != 2 or var.domain[0] < 0 or var.domain[1] > 1:
            raise TypeError('TypeError: ' + str(x) + ' is not a boolean variable')
    elif not isinstance(x, _NotBooleanVariable):
        raise TypeError('TypeError: ' + str(x) + ' is not a boolean variable')