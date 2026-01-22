from typing import Callable
from scipy.sparse import csr_matrix
from pennylane import math
from pennylane.ops import Sum, Hamiltonian
from pennylane.measurements import (
from pennylane.pauli.conversion import is_pauli_sentence, pauli_sentence
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .apply_operation import apply_operation
def csr_dot_products(measurementprocess: ExpectationMP, state: TensorLike, is_state_batched: bool=False) -> TensorLike:
    """Measure the expectation value of an observable using dot products between ``scipy.csr_matrix``
    representations.

    Args:
        measurementprocess (ExpectationMP): measurement process to apply to the state
        state (TensorLike): the state to measure
        is_state_batched (bool): whether the state is batched or not

    Returns:
        TensorLike: the result of the measurement
    """
    total_wires = len(state.shape) - is_state_batched
    if is_pauli_sentence(measurementprocess.obs):
        state = math.toarray(state)
        if is_state_batched:
            state = state.reshape(math.shape(state)[0], -1)
        else:
            state = state.reshape(1, -1)
        bra = math.conj(state)
        ps = pauli_sentence(measurementprocess.obs)
        new_ket = ps.dot(state, wire_order=list(range(total_wires)))
        res = (bra * new_ket).sum(axis=1)
    elif is_state_batched:
        Hmat = measurementprocess.obs.sparse_matrix(wire_order=list(range(total_wires)))
        state = math.toarray(state).reshape(math.shape(state)[0], -1)
        bra = csr_matrix(math.conj(state))
        ket = csr_matrix(state)
        new_bra = bra.dot(Hmat)
        res = new_bra.multiply(ket).sum(axis=1).getA()
    else:
        Hmat = measurementprocess.obs.sparse_matrix(wire_order=list(range(total_wires)))
        state = math.toarray(state).flatten()
        bra = csr_matrix(math.conj(state))
        ket = csr_matrix(state[..., None])
        new_ket = csr_matrix.dot(Hmat, ket)
        res = csr_matrix.dot(bra, new_ket).toarray()[0]
    return math.real(math.squeeze(res))