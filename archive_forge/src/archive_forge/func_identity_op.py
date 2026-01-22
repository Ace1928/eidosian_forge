from functools import lru_cache
from scipy import sparse
import pennylane as qml
from pennylane.operation import AnyWires, AllWires, CVObservable, Operation
@staticmethod
def identity_op(*params):
    """Alias for matrix representation of the identity operator."""
    return I.compute_matrix(*params)