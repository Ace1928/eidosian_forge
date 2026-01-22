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
@classmethod
def weighted_sum(cls, expressions, coefficients):
    """Creates the expression sum(expressions[i] * coefficients[i])."""
    if LinearExpr.is_empty_or_all_null(coefficients):
        return 0
    elif len(expressions) == 1:
        return expressions[0] * coefficients[0]
    else:
        return _WeightedSum(expressions, coefficients)