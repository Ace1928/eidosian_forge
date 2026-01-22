import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def generate_library_of_2q_circuits(n_library_circuits: int, two_qubit_gate: 'cirq.Gate', *, max_cycle_depth: int=100, q0: 'cirq.Qid'=devices.LineQubit(0), q1: 'cirq.Qid'=devices.LineQubit(1), random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List['cirq.Circuit']:
    """Generate a library of two-qubit Circuits.

    For single-qubit gates, this uses PhasedXZGates where the axis-in-XY-plane is one
    of eight eighth turns and the Z rotation angle is one of eight eighth turns. This
    provides 8*8=64 total choices, each implementable with one PhasedXZGate. This is
    appropriate for architectures with microwave single-qubit control.

    Args:
        n_library_circuits: The number of circuits to generate.
        two_qubit_gate: The two qubit gate to use in the circuits.
        max_cycle_depth: The maximum cycle_depth in the circuits to generate. If you are using XEB,
            this must be greater than or equal to the maximum value in `cycle_depths`.
        q0: The first qubit to use when constructing the circuits.
        q1: The second qubit to use when constructing the circuits
        random_state: A random state or seed used to deterministically sample the random circuits.
    """
    rs = value.parse_random_state(random_state)
    exponents = np.linspace(0, 7 / 4, 8)
    single_qubit_gates = [ops.PhasedXZGate(x_exponent=0.5, z_exponent=z, axis_phase_exponent=a) for a, z in itertools.product(exponents, repeat=2)]
    return [random_rotations_between_two_qubit_circuit(q0, q1, depth=max_cycle_depth, two_qubit_op_factory=lambda a, b, _: two_qubit_gate(a, b), single_qubit_gates=single_qubit_gates, seed=rs) for _ in range(n_library_circuits)]