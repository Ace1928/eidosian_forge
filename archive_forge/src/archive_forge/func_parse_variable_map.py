from typing import Dict, FrozenSet, Generic, Iterable, Mapping, Optional, Set, TypeVar
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model
def parse_variable_map(proto: sparse_containers_pb2.SparseDoubleVectorProto, mod: model.Model) -> Dict[model.Variable, float]:
    """Converts a sparse vector of variables from proto to dict representation."""
    result = {}
    for index, var_id in enumerate(proto.ids):
        result[mod.get_variable(var_id)] = proto.values[index]
    return result