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
def xx_circuit_step(source, strength, target, embodiment):
    """
    Builds a single step in an XX-based circuit.

    `source` and `target` are positive canonical coordinates; `strength` is the interaction strength
    at this step in the circuit as a canonical coordinate (so that CX = RZX(pi/2) corresponds to
    pi/4); and `embodiment` is a Qiskit circuit which enacts the canonical gate of the prescribed
    interaction `strength`.
    """
    permute_source_for_overlap, permute_target_for_overlap = (None, None)
    for source_reflection_name in reflection_options:
        reflected_source_coord, source_reflection, reflection_phase_shift = apply_reflection(source_reflection_name, source)
        for source_shift_name in shift_options:
            shifted_source_coord, source_shift, shift_phase_shift = apply_shift(source_shift_name, reflected_source_coord)
            source_shared, target_shared = (None, None)
            for i, j in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
                if abs(np.mod(abs(shifted_source_coord[i] - target[j]), np.pi)) < EPSILON or abs(np.mod(abs(shifted_source_coord[i] - target[j]), np.pi) - np.pi) < EPSILON:
                    source_shared, target_shared = (i, j)
                    break
            if source_shared is None:
                continue
            source_first, source_second = (x for x in [0, 1, 2] if x != source_shared)
            target_first, target_second = (x for x in [0, 1, 2] if x != target_shared)
            r, s, u, v, x, y = decompose_xxyy_into_xxyy_xx(float(target[target_first]), float(target[target_second]), float(shifted_source_coord[source_first]), float(shifted_source_coord[source_second]), float(strength))
            if any((math.isnan(val) for val in (r, s, u, v, x, y))):
                continue
            permute_source_for_overlap = canonical_rotation_circuit(source_first, source_second)
            permute_target_for_overlap = canonical_rotation_circuit(target_first, target_second)
            break
        if permute_source_for_overlap is not None:
            break
    if permute_source_for_overlap is None:
        raise QiskitError(f'Error during RZX decomposition: Could not find a suitable Weyl reflection to match {source} to {target} along {strength}.')
    prefix_circuit, affix_circuit = (QuantumCircuit(2), QuantumCircuit(2))
    prefix_circuit.compose(permute_target_for_overlap.inverse(), inplace=True)
    prefix_circuit.rz(2 * x, [0])
    prefix_circuit.rz(2 * y, [1])
    prefix_circuit.compose(embodiment, inplace=True)
    prefix_circuit.rz(2 * u, [0])
    prefix_circuit.rz(2 * v, [1])
    prefix_circuit.compose(permute_source_for_overlap, inplace=True)
    prefix_circuit.compose(source_reflection, inplace=True)
    prefix_circuit.global_phase += -np.log(reflection_phase_shift).imag
    prefix_circuit.global_phase += -np.log(shift_phase_shift).imag
    affix_circuit.compose(source_reflection.inverse(), inplace=True)
    affix_circuit.compose(source_shift, inplace=True)
    affix_circuit.compose(permute_source_for_overlap.inverse(), inplace=True)
    affix_circuit.rz(2 * r, [0])
    affix_circuit.rz(2 * s, [1])
    affix_circuit.compose(permute_target_for_overlap, inplace=True)
    return {'prefix_circuit': prefix_circuit, 'affix_circuit': affix_circuit}