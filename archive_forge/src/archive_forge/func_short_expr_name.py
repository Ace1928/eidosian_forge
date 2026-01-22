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
def short_expr_name(model: cp_model_pb2.CpModelProto, e: cp_model_pb2.LinearExpressionProto) -> str:
    """Pretty-print LinearExpressionProto instances."""
    if not e.vars:
        return str(e.offset)
    if len(e.vars) == 1:
        var_name = short_name(model, e.vars[0])
        coeff = e.coeffs[0]
        result = ''
        if coeff == 1:
            result = var_name
        elif coeff == -1:
            result = f'-{var_name}'
        elif coeff != 0:
            result = f'{coeff} * {var_name}'
        if e.offset > 0:
            result = f'{result} + {e.offset}'
        elif e.offset < 0:
            result = f'{result} - {-e.offset}'
        return result
    return str(e)