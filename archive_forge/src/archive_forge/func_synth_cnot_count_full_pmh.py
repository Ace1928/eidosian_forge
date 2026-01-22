from __future__ import annotations
import copy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
def synth_cnot_count_full_pmh(state: list[list[bool]] | np.ndarray[bool], section_size: int=2) -> QuantumCircuit:
    """
    Synthesize linear reversible circuits for all-to-all architecture
    using Patel, Markov and Hayes method.

    This function is an implementation of the Patel, Markov and Hayes algorithm from [1]
    for optimal synthesis of linear reversible circuits for all-to-all architecture,
    as specified by an :math:`n \\times n` matrix.

    Args:
        state: :math:`n \\times n` boolean invertible matrix, describing
            the state of the input circuit
        section_size: The size of each section, used in the
            Patel–Markov–Hayes algorithm [1]. ``section_size`` must be a factor of the number
            of qubits.

    Returns:
        QuantumCircuit: a CX-only circuit implementing the linear transformation.

    Raises:
        QiskitError: when variable ``state`` isn't of type ``numpy.ndarray``

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """
    if not isinstance(state, (list, np.ndarray)):
        raise QiskitError('state should be of type list or numpy.ndarray, but was of the type {}'.format(type(state)))
    state = np.array(state)
    [state, circuit_l] = _lwr_cnot_synth(state, section_size)
    state = np.transpose(state)
    [state, circuit_u] = _lwr_cnot_synth(state, section_size)
    circuit_l.reverse()
    for i in circuit_u:
        i.reverse()
    circ = QuantumCircuit(state.shape[0])
    for i in circuit_u + circuit_l:
        circ.cx(i[0], i[1])
    return circ