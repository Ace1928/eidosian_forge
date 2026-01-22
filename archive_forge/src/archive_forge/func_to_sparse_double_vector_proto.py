from typing import Dict, FrozenSet, Generic, Iterable, Mapping, Optional, Set, TypeVar
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model
def to_sparse_double_vector_proto(terms: Mapping[VarOrConstraintType, float]) -> sparse_containers_pb2.SparseDoubleVectorProto:
    """Converts a sparse vector from proto to dict representation."""
    result = sparse_containers_pb2.SparseDoubleVectorProto()
    if terms:
        id_and_values = [(key.id, value) for key, value in terms.items()]
        id_and_values.sort()
        ids, values = zip(*id_and_values)
        result.ids[:] = ids
        result.values[:] = values
    return result