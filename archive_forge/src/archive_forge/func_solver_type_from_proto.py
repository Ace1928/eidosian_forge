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
def solver_type_from_proto(proto_value: math_opt_parameters_pb2.SolverTypeProto) -> Optional[SolverType]:
    if proto_value == math_opt_parameters_pb2.SOLVER_TYPE_UNSPECIFIED:
        return None
    return SolverType(proto_value)