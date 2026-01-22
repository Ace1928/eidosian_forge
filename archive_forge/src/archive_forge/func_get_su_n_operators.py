import warnings
from typing import Sequence, Callable
import numpy as np
from scipy.sparse.linalg import expm
import pennylane as qml
from pennylane import transform
from pennylane.tape import QuantumTape
from pennylane.queuing import QueuingManager
def get_su_n_operators(self, restriction):
    """Get the SU(N) operators. The dimension of the group is :math:`N^2-1`.

        Args:
            restriction (.Hamiltonian): Restrict the Riemannian gradient to a subspace.

        Returns:
            tuple[list[array[complex]], list[str]]: list of :math:`N^2 \\times N^2` NumPy complex arrays and corresponding Pauli words.
        """
    operators = []
    names = []
    wire_map = dict(zip(range(self.nqubits), range(self.nqubits)))
    if restriction is None:
        for ps in qml.pauli.pauli_group(self.nqubits):
            operators.append(ps)
            names.append(qml.pauli.pauli_word_to_string(ps, wire_map=wire_map))
    else:
        for ps in set(restriction.ops):
            operators.append(ps)
            names.append(qml.pauli.pauli_word_to_string(ps, wire_map=wire_map))
    return (operators, names)