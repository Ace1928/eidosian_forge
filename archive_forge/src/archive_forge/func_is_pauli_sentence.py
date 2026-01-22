from functools import reduce, singledispatch
from itertools import product
from operator import matmul
from typing import Union, Tuple
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum
from pennylane.ops.qubit.matrix_ops import _walsh_hadamard_transform
from .pauli_arithmetic import I, PauliSentence, PauliWord, X, Y, Z, op_map
from .utils import is_pauli_word
def is_pauli_sentence(op):
    """Returns True of the operator is a PauliSentence and False otherwise."""
    if op.pauli_rep is not None:
        return True
    if isinstance(op, Hamiltonian):
        return all((is_pauli_word(o) for o in op.ops))
    return False