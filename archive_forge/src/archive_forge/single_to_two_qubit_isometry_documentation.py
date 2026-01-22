from typing import List, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.transformers.analytical_decompositions import two_qubit_to_cz
Decomposes a 2q operation into at-most 2 CZs + 1q rotations; assuming `q0` is initially |0>.

    The method implements isometry from one to two qubits; assuming qubit `q0` is always in the |0>
    state. See Appendix B.1 of https://arxiv.org/abs/1501.06911 for more details.

    Args:
        q0: The first qubit being operated on. This is assumed to always be in the |0> state.
        q1: The other qubit being operated on.
        mat: Defines the unitary operation to apply to the pair of qubits.
        allow_partial_czs: Enables the use of Partial-CZ gates.
        atol: A limit on the amount of absolute error introduced by the construction.
        clean_operations: Enables optimizing resulting operation list by merging single qubit
        operations and ejecting phased Paulis and Z operations.

    Returns:
        A list of operations implementing the action of the given unitary matrix, assuming
        the input qubit `q0` is in the |0> state.
    