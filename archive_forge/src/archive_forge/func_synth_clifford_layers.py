from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford  # pylint: disable=cyclic-import
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from qiskit.synthesis.linear import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my
from qiskit.synthesis.linear.linear_matrix_utils import (
def synth_clifford_layers(cliff: Clifford, cx_synth_func: Callable[[np.ndarray], QuantumCircuit]=_default_cx_synth_func, cz_synth_func: Callable[[np.ndarray], QuantumCircuit]=_default_cz_synth_func, cx_cz_synth_func: Callable[[np.ndarray], QuantumCircuit] | None=None, cz_func_reverse_qubits: bool=False, validate: bool=False) -> QuantumCircuit:
    """Synthesis of a :class:`.Clifford` into layers, it provides a similar
    decomposition to the synthesis described in Lemma 8 of Bravyi and Maslov [1].

    For example, a 5-qubit Clifford circuit is decomposed into the following layers:

    .. parsed-literal::
             ┌─────┐┌─────┐┌────────┐┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐
        q_0: ┤0    ├┤0    ├┤0       ├┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_1: ┤1    ├┤1    ├┤1       ├┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_2: ┤2 S2 ├┤2 CZ ├┤2 CX_dg ├┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_3: ┤3    ├┤3    ├┤3       ├┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_4: ┤4    ├┤4    ├┤4       ├┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├
             └─────┘└─────┘└────────┘└─────┘└─────┘└─────┘└─────┘└────────┘

    This decomposition is for the default ``cz_synth_func`` and ``cx_synth_func`` functions,
    with other functions one may see slightly different decomposition.

    Args:
        cliff: A Clifford operator.
        cx_synth_func: A function to decompose the CX sub-circuit.
            It gets as input a boolean invertible matrix, and outputs a :class:`.QuantumCircuit`.
        cz_synth_func: A function to decompose the CZ sub-circuit.
            It gets as input a boolean symmetric matrix, and outputs a :class:`.QuantumCircuit`.
        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.
        validate (Boolean): if True, validates the synthesis process.
        cz_func_reverse_qubits (Boolean): True only if ``cz_synth_func`` is
            :func:`.synth_cz_depth_line_mr`, since this function returns a circuit that reverts
            the order of qubits.

    Returns:
        A circuit implementation of the Clifford.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    num_qubits = cliff.num_qubits
    if cz_func_reverse_qubits:
        cliff0 = _reverse_clifford(cliff)
    else:
        cliff0 = cliff
    qubit_list = list(range(num_qubits))
    layeredCircuit = QuantumCircuit(num_qubits)
    H1_circ, cliff1 = _create_graph_state(cliff0, validate=validate)
    H2_circ, CZ1_circ, S1_circ, cliff2 = _decompose_graph_state(cliff1, validate=validate, cz_synth_func=cz_synth_func)
    S2_circ, CZ2_circ, CX_circ = _decompose_hadamard_free(cliff2.adjoint(), validate=validate, cz_synth_func=cz_synth_func, cx_synth_func=cx_synth_func, cx_cz_synth_func=cx_cz_synth_func, cz_func_reverse_qubits=cz_func_reverse_qubits)
    layeredCircuit.append(S2_circ, qubit_list)
    if cx_cz_synth_func is None:
        layeredCircuit.append(CZ2_circ, qubit_list)
        CXinv = CX_circ.copy().inverse()
        layeredCircuit.append(CXinv, qubit_list)
    else:
        layeredCircuit.append(CX_circ, qubit_list)
    layeredCircuit.append(H2_circ, qubit_list)
    layeredCircuit.append(S1_circ, qubit_list)
    layeredCircuit.append(CZ1_circ, qubit_list)
    if cz_func_reverse_qubits:
        H1_circ = H1_circ.reverse_bits()
    layeredCircuit.append(H1_circ, qubit_list)
    clifford_target = Clifford(layeredCircuit)
    pauli_circ = _calc_pauli_diff(cliff, clifford_target)
    layeredCircuit.append(pauli_circ, qubit_list)
    return layeredCircuit