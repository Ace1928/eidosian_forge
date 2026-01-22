import logging
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, Union
import numpy as np
from scipy.sparse import dok_matrix
import cvxpy.settings as s
from cvxpy import Zero
from cvxpy.constraints import SOC, ExpCone, NonNeg
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def get_variable_type(n: int, data: Dict[str, Any]) -> str:
    """Given an index n, and a set of data,
    return the type of a variable with the same index."""
    if n in data[s.BOOL_IDX]:
        return VariableTypes.BINARY
    elif n in data[s.INT_IDX]:
        return VariableTypes.INTEGER
    return VariableTypes.CONTINUOUS