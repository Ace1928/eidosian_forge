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
def term_2_pauli_word(term):
    if isinstance(term, Tensor):
        pw = dict(((obs.wires[0], obs.name[-1]) for obs in term.non_identity_obs))
    elif isinstance(term, Identity):
        pw = {}
    else:
        pw = dict([(term.wires[0], term.name[-1])])
    return PauliWord(pw)