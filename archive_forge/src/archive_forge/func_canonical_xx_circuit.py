from __future__ import annotations
from functools import reduce
import math
from operator import itemgetter
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXXGate, RYYGate, RZGate
from qiskit.exceptions import QiskitError
from .paths import decomposition_hop
from .utilities import EPSILON, safe_arccos
from .weyl import (
def canonical_xx_circuit(target, strength_sequence, basis_embodiments):
    """
    Assembles a Qiskit circuit from a specified `strength_sequence` of XX-type interactions which
    emulates the canonical gate at canonical coordinate `target`.  The circuits supplied by
    `basis_embodiments` are used to instantiate the individual XX actions.

    NOTE: The elements of `strength_sequence` are expected to be normalized so that np.pi/2
        corresponds to RZX(np.pi/2) = CX; `target` is taken to be a positive canonical coordinate;
        and `basis_embodiments` maps `strength_sequence` elements to circuits which instantiate
        these gates.
    """
    if len(strength_sequence) == 0:
        return QuantumCircuit(2)
    prefix_circuit, affix_circuit = (QuantumCircuit(2), QuantumCircuit(2))
    while len(strength_sequence) > 1:
        source = decomposition_hop(target, strength_sequence)
        strength = strength_sequence[-1]
        preceding_prefix_circuit, preceding_affix_circuit = itemgetter('prefix_circuit', 'affix_circuit')(xx_circuit_step(source, strength / 2, target, basis_embodiments[strength]))
        prefix_circuit.compose(preceding_prefix_circuit, inplace=True)
        affix_circuit.compose(preceding_affix_circuit, inplace=True, front=True)
        target, strength_sequence = (source, strength_sequence[:-1])
    circuit = prefix_circuit
    if target[0] <= np.pi / 4:
        circuit.compose(basis_embodiments[strength_sequence[0]], inplace=True)
    else:
        _, source_reflection, reflection_phase_shift = apply_reflection('reflect XX, YY', [0, 0, 0])
        _, source_shift, shift_phase_shift = apply_shift('X shift', [0, 0, 0])
        circuit.compose(source_reflection, inplace=True)
        circuit.compose(basis_embodiments[strength_sequence[0]], inplace=True)
        circuit.compose(source_reflection.inverse(), inplace=True)
        circuit.compose(source_shift, inplace=True)
        circuit.global_phase += -np.log(shift_phase_shift).imag
        circuit.global_phase += -np.log(reflection_phase_shift).imag
    circuit.compose(affix_circuit, inplace=True)
    return circuit