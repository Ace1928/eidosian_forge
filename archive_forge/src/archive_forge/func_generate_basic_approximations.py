from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals
from .gate_sequence import GateSequence
def generate_basic_approximations(basis_gates: list[str | Gate], depth: int, filename: str | None=None) -> list[GateSequence]:
    """Generates a list of :class:`GateSequence`\\ s with the gates in ``basis_gates``.

    Args:
        basis_gates: The gates from which to create the sequences of gates.
        depth: The maximum depth of the approximations.
        filename: If provided, the basic approximations are stored in this file.

    Returns:
        List of :class:`GateSequence`\\ s using the gates in ``basis_gates``.

    Raises:
        ValueError: If ``basis_gates`` contains an invalid gate identifier.
    """
    basis = []
    for gate in basis_gates:
        if isinstance(gate, str):
            if gate not in _1q_gates.keys():
                raise ValueError(f'Invalid gate identifier: {gate}')
            basis.append(gate)
        else:
            basis.append(gate.name)
    tree = Node((), GateSequence(), [])
    cur_level = [tree]
    sequences = [tree.sequence]
    for _ in [None] * depth:
        next_level = []
        for node in cur_level:
            next_level.extend(_process_node(node, basis, sequences))
        cur_level = next_level
    if filename is not None:
        data = {}
        for sequence in sequences:
            gatestring = sequence.name
            data[gatestring] = sequence.product
        np.save(filename, data)
    return sequences