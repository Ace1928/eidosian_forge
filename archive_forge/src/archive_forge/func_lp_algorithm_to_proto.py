import dataclasses
import datetime
import enum
from typing import Dict, Optional
from ortools.pdlp import solvers_pb2 as pdlp_solvers_pb2
from ortools.glop import parameters_pb2 as glop_parameters_pb2
from ortools.gscip import gscip_pb2
from ortools.math_opt import parameters_pb2 as math_opt_parameters_pb2
from ortools.math_opt.solvers import glpk_pb2
from ortools.math_opt.solvers import gurobi_pb2
from ortools.math_opt.solvers import highs_pb2
from ortools.math_opt.solvers import osqp_pb2
from ortools.sat import sat_parameters_pb2
def lp_algorithm_to_proto(lp_algorithm: Optional[LPAlgorithm]) -> math_opt_parameters_pb2.LPAlgorithmProto:
    if lp_algorithm is None:
        return math_opt_parameters_pb2.LP_ALGORITHM_UNSPECIFIED
    return lp_algorithm.value