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
Adds SOC constraint to the model using the data from mat and vec.

        Return tuple contains (QConstr, list of Constr, and list of variables).
        