import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from .clifford_decompose_bm import _decompose_clifford_1q
def synth_clifford_ag(clifford: Clifford) -> QuantumCircuit:
    """Decompose a :class:`.Clifford` operator into a :class:`.QuantumCircuit`
    based on Aaronson-Gottesman method [1].

    Args:
        clifford: A Clifford operator.

    Returns:
        A circuit implementation of the Clifford.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """
    if clifford.num_qubits == 1:
        return _decompose_clifford_1q(clifford.tableau)
    circuit = QuantumCircuit(clifford.num_qubits, name=str(clifford))
    clifford_cpy = clifford.copy()
    for i in range(clifford.num_qubits):
        _set_qubit_x_true(clifford_cpy, circuit, i)
        _set_row_x_zero(clifford_cpy, circuit, i)
        _set_row_z_zero(clifford_cpy, circuit, i)
    for i in range(clifford.num_qubits):
        if clifford_cpy.destab_phase[i]:
            _append_z(clifford_cpy, i)
            circuit.z(i)
        if clifford_cpy.stab_phase[i]:
            _append_x(clifford_cpy, i)
            circuit.x(i)
    return circuit.inverse()