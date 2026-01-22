from typing import Dict, FrozenSet, Generic, Iterable, Mapping, Optional, Set, TypeVar
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model
def parse_linear_constraint_map(proto: sparse_containers_pb2.SparseDoubleVectorProto, mod: model.Model) -> Dict[model.LinearConstraint, float]:
    """Converts a sparse vector of linear constraints from proto to dict representation."""
    result = {}
    for index, lin_con_id in enumerate(proto.ids):
        result[mod.get_linear_constraint(lin_con_id)] = proto.values[index]
    return result