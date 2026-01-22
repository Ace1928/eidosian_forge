from copy import copy
import numpy as np
import pennylane as qml
from pennylane.ops import Prod, SProd
from pennylane.pauli.utils import (
from pennylane.wires import Wires
from .graph_colouring import largest_first, recursive_largest_first
def complement_adj_matrix_for_operator(self):
    """Constructs the adjacency matrix for the complement of the Pauli graph.

        The adjacency matrix for an undirected graph of N vertices is an N by N symmetric binary
        matrix, where matrix elements of 1 denote an edge, and matrix elements of 0 denote no edge.

        Returns:
            array[int]: the square and symmetric adjacency matrix
        """
    if self.binary_observables is None:
        self.binary_observables = self.binary_repr()
    n_qubits = int(np.shape(self.binary_observables)[1] / 2)
    if self.grouping_type == 'qwc':
        adj = qwc_complement_adj_matrix(self.binary_observables)
    elif self.grouping_type in frozenset(['commuting', 'anticommuting']):
        symplectic_form = np.block([[np.zeros((n_qubits, n_qubits)), np.eye(n_qubits)], [np.eye(n_qubits), np.zeros((n_qubits, n_qubits))]])
        mat_prod = self.binary_observables @ symplectic_form @ np.transpose(self.binary_observables)
        if self.grouping_type == 'commuting':
            adj = mat_prod % 2
        elif self.grouping_type == 'anticommuting':
            adj = (mat_prod + 1) % 2
            np.fill_diagonal(adj, 0)
    return adj