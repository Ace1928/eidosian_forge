import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
def parse_problem_status(proto: result_pb2.ProblemStatusProto) -> ProblemStatus:
    """Returns an equivalent ProblemStatus from the input proto."""
    primal_status_proto = proto.primal_status
    if primal_status_proto == result_pb2.FEASIBILITY_STATUS_UNSPECIFIED:
        raise ValueError('Primal feasibility status should not be UNSPECIFIED')
    dual_status_proto = proto.dual_status
    if dual_status_proto == result_pb2.FEASIBILITY_STATUS_UNSPECIFIED:
        raise ValueError('Dual feasibility status should not be UNSPECIFIED')
    return ProblemStatus(primal_status=FeasibilityStatus(primal_status_proto), dual_status=FeasibilityStatus(dual_status_proto), primal_or_dual_infeasible=proto.primal_or_dual_infeasible)