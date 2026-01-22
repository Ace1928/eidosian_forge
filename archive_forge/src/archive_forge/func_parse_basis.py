import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def parse_basis(proto: solution_pb2.BasisProto, mod: model.Model) -> Basis:
    """Returns an equivalent Basis to the input proto."""
    result = Basis()
    for index, vid in enumerate(proto.variable_status.ids):
        status_proto = proto.variable_status.values[index]
        if status_proto == solution_pb2.BASIS_STATUS_UNSPECIFIED:
            raise ValueError('Variable basis status should not be UNSPECIFIED')
        result.variable_status[mod.get_variable(vid)] = BasisStatus(status_proto)
    for index, cid in enumerate(proto.constraint_status.ids):
        status_proto = proto.constraint_status.values[index]
        if status_proto == solution_pb2.BASIS_STATUS_UNSPECIFIED:
            raise ValueError('Constraint basis status should not be UNSPECIFIED')
        result.constraint_status[mod.get_linear_constraint(cid)] = BasisStatus(status_proto)
    status_proto = proto.basic_dual_feasibility
    if status_proto == solution_pb2.SOLUTION_STATUS_UNSPECIFIED:
        raise ValueError('Basic dual feasibility status should not be UNSPECIFIED')
    result.basic_dual_feasibility = SolutionStatus(status_proto)
    return result