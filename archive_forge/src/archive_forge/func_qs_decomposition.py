from __future__ import annotations
from typing import Callable
import scipy
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.synthesis.two_qubit import (
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library.generalized_gates.uc_pauli_rot import UCPauliRotGate, _EPS
from qiskit.circuit.library.generalized_gates.ucry import UCRYGate
from qiskit.circuit.library.generalized_gates.ucrz import UCRZGate
def qs_decomposition(mat: np.ndarray, opt_a1: bool=True, opt_a2: bool=True, decomposer_1q: Callable[[np.ndarray], QuantumCircuit] | None=None, decomposer_2q: Callable[[np.ndarray], QuantumCircuit] | None=None, *, _depth=0):
    """
    Decomposes a unitary matrix into one and two qubit gates using Quantum Shannon Decomposition,

    This decomposition is described in Shende et al. [1].

    .. parsed-literal::
          ┌───┐               ┌───┐     ┌───┐     ┌───┐
         ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────
          │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐
        /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├
          └───┘          └───┘     └───┘     └───┘     └───┘

    The number of :class:`.CXGate`\\ s generated with the decomposition without optimizations is:

    .. math::

        \\frac{9}{16} 4^n - \\frac{3}{2} 2^n

    If ``opt_a1 = True``, the default, the CX count is reduced by:

    .. math::

        \\frac{1}{3} 4^{n - 2} - 1.

    If ``opt_a2 = True``, the default, the CX count is reduced by:

    .. math::

        4^{n-2} - 1.

    Args:
        mat: unitary matrix to decompose
        opt_a1: whether to try optimization A.1 from Shende et al. [1].
            This should eliminate 1 ``cx`` per call.
            If ``True``, :class:`.CZGate`\\s are left in the output.
            If desired these can be further decomposed to :class:`.CXGate`\\s.
        opt_a2: whether to try optimization A.2 from Shende et al. [1].
            This decomposes two qubit unitaries into a diagonal gate and
            a two cx unitary and reduces overall cx count by :math:`4^{n-2} - 1`.
        decomposer_1q: optional 1Q decomposer. If None, uses
            :class:`~qiskit.synthesis.OneQubitEulerDecomposer`.
        decomposer_2q: optional 2Q decomposer. If None, uses
            :class:`~qiskit.synthesis.TwoQubitBasisDecomposer`.

    Returns:
        QuantumCircuit: Decomposed quantum circuit.

    References:
        1. Shende, Bullock, Markov, *Synthesis of Quantum Logic Circuits*,
           `arXiv:0406176 [quant-ph] <https://arxiv.org/abs/quant-ph/0406176>`_
    """
    dim = mat.shape[0]
    nqubits = int(np.log2(dim))
    if np.allclose(np.identity(dim), mat):
        return QuantumCircuit(nqubits)
    if dim == 2:
        if decomposer_1q is None:
            decomposer_1q = one_qubit_decompose.OneQubitEulerDecomposer()
        circ = decomposer_1q(mat)
    elif dim == 4:
        if decomposer_2q is None:
            if opt_a2 and _depth > 0:
                from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

                def decomp_2q(mat):
                    ugate = UnitaryGate(mat)
                    qc = QuantumCircuit(2, name='qsd2q')
                    qc.append(ugate, [0, 1])
                    return qc
                decomposer_2q = decomp_2q
            else:
                decomposer_2q = TwoQubitBasisDecomposer(CXGate())
        circ = decomposer_2q(mat)
    else:
        qr = QuantumRegister(nqubits)
        circ = QuantumCircuit(qr)
        dim_o2 = dim // 2
        (u1, u2), vtheta, (v1h, v2h) = scipy.linalg.cossin(mat, separate=True, p=dim_o2, q=dim_o2)
        left_circ = _demultiplex(v1h, v2h, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth)
        circ.append(left_circ.to_instruction(), qr)
        if opt_a1:
            nangles = len(vtheta)
            half_size = nangles // 2
            circ_cz = _get_ucry_cz(nqubits, (2 * vtheta).tolist())
            circ.append(circ_cz.to_instruction(), range(nqubits))
            u2[:, half_size:] = np.negative(u2[:, half_size:])
        else:
            ucry = UCRYGate((2 * vtheta).tolist())
            circ.append(ucry, [qr[-1]] + qr[:-1])
        right_circ = _demultiplex(u1, u2, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth)
        circ.append(right_circ.to_instruction(), qr)
    if opt_a2 and _depth == 0 and (dim > 4):
        return _apply_a2(circ)
    return circ