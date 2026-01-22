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
def object_is_a_false_literal(literal: LiteralT) -> bool:
    """Checks if literal is either False, or a Boolean literals fixed to False."""
    if isinstance(literal, IntVar):
        proto = literal.proto
        return len(proto.domain) == 2 and proto.domain[0] == 0 and (proto.domain[1] == 0)
    if isinstance(literal, _NotBooleanVariable):
        proto = literal.negated().proto
        return len(proto.domain) == 2 and proto.domain[0] == 1 and (proto.domain[1] == 1)
    if isinstance(literal, numbers.Integral):
        return int(literal) == 0
    return False