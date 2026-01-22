from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.operators.channel import SuperOp
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT
def schmidt_decomposition(state, qargs):
    """Return the Schmidt Decomposition of a pure quantum state.

    For an arbitrary bipartite state:

    .. math::
         |\\psi\\rangle_{AB} = \\sum_{i,j} c_{ij}
                             |x_i\\rangle_A \\otimes |y_j\\rangle_B,

    its Schmidt Decomposition is given by the single-index sum over k:

    .. math::
        |\\psi\\rangle_{AB} = \\sum_{k} \\lambda_{k}
                            |u_k\\rangle_A \\otimes |v_k\\rangle_B

    where :math:`|u_k\\rangle_A` and :math:`|v_k\\rangle_B` are an
    orthonormal set of vectors in their respective spaces :math:`A` and :math:`B`,
    and the Schmidt coefficients :math:`\\lambda_k` are positive real values.

    Args:
        state (Statevector or DensityMatrix): the input state.
        qargs (list): the list of Input state positions corresponding to subsystem :math:`B`.

    Returns:
        list: list of tuples ``(s, u, v)``, where ``s`` (float) are the Schmidt coefficients
        :math:`\\lambda_k`, and ``u`` (Statevector), ``v`` (Statevector) are the Schmidt vectors
        :math:`|u_k\\rangle_A`, :math:`|u_k\\rangle_B`, respectively.

    Raises:
        QiskitError: if Input qargs is not a list of positions of the Input state.
        QiskitError: if Input qargs is not a proper subset of Input state.

    .. note::
        In Qiskit, qubits are ordered using little-endian notation, with the least significant
        qubits having smaller indices. For example, a four-qubit system is represented as
        :math:`|q_3q_2q_1q_0\\rangle`. Using this convention, setting ``qargs=[0]`` will partition the
        state as :math:`|q_3q_2q_1\\rangle_A\\otimes|q_0\\rangle_B`. Furthermore, qubits will be organized
        in this notation regardless of the order they are passed. For instance, passing either
        ``qargs=[1,2]`` or ``qargs=[2,1]`` will result in partitioning the state as
        :math:`|q_3q_0\\rangle_A\\otimes|q_2q_1\\rangle_B`.
    """
    state = _format_state(state, validate=False)
    if isinstance(state, DensityMatrix):
        state = state.to_statevector()
    dims = state.dims()
    state_tens = state._data.reshape(dims[::-1])
    ndim = state_tens.ndim
    qudits = list(range(ndim))
    if not isinstance(qargs, (list, np.ndarray)):
        raise QiskitError('Input qargs is not a list of positions of the Input state')
    qargs = set(qargs)
    if qargs == set(qudits) or not qargs.issubset(qudits):
        raise QiskitError('Input qargs is not a proper subset of Input state')
    qargs_b = list(qargs)
    qargs_a = [i for i in qudits if i not in qargs_b]
    dims_b = state.dims(qargs_b)
    dims_a = state.dims(qargs_a)
    ndim_b = np.prod(dims_b)
    ndim_a = np.prod(dims_a)
    qargs_axes = [qudits[::-1].index(i) for i in qargs_b + qargs_a][::-1]
    state_tens = state_tens.transpose(qargs_axes)
    state_mat = state_tens.reshape([ndim_a, ndim_b])
    u_mat, s_arr, vh_mat = np.linalg.svd(state_mat, full_matrices=False)
    schmidt_components = [(s, Statevector(u, dims=dims_a), Statevector(v, dims=dims_b)) for s, u, v in zip(s_arr, u_mat.T, vh_mat) if s > ATOL_DEFAULT]
    return schmidt_components