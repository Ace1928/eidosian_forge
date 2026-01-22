from __future__ import annotations
import numpy as np
from qiskit.circuit.gate import Gate
from .gate_sequence import GateSequence
from .commutator_decompose import commutator_decompose
from .generate_basis_approximations import generate_basic_approximations, _1q_gates, _1q_inverses
def load_basic_approximations(self, data: list | str | dict) -> list[GateSequence]:
    """Load basic approximations.

        Args:
            data: If a string, specifies the path to the file from where to load the data.
                If a dictionary, directly specifies the decompositions as ``{gates: matrix}``.
                There ``gates`` are the names of the gates producing the SO(3) matrix ``matrix``,
                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.

        Returns:
            A list of basic approximations as type ``GateSequence``.

        Raises:
            ValueError: If the number of gate combinations and associated matrices does not match.
        """
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        data = np.load(data, allow_pickle=True)
    sequences = []
    for gatestring, matrix in data.items():
        sequence = GateSequence()
        sequence.gates = [_1q_gates[element] for element in gatestring.split()]
        sequence.product = np.asarray(matrix)
        sequences.append(sequence)
    return sequences