import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
def parse_primal_ray(proto: solution_pb2.PrimalRayProto, mod: model.Model) -> PrimalRay:
    """Returns an equivalent PrimalRay from the input proto."""
    result = PrimalRay()
    result.variable_values = sparse_containers.parse_variable_map(proto.variable_values, mod)
    return result