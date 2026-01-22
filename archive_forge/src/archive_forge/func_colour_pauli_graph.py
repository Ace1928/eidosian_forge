from copy import copy
import numpy as np
import pennylane as qml
from pennylane.ops import Prod, SProd
from pennylane.pauli.utils import (
from pennylane.wires import Wires
from .graph_colouring import largest_first, recursive_largest_first
def colour_pauli_graph(self):
    """
        Runs the graph colouring heuristic algorithm to obtain the partitioned Pauli words.

        Returns:
            list[list[Observable]]: a list of the obtained groupings. Each grouping is itself a
            list of Pauli word ``Observable`` instances
        """
    if self.adj_matrix is None:
        self.adj_matrix = self.complement_adj_matrix_for_operator()
    coloured_binary_paulis = self.graph_colourer(self.binary_observables, self.adj_matrix)
    self.grouped_paulis = [[binary_to_pauli(pauli_word, wire_map=self._wire_map) for pauli_word in grouping] for grouping in coloured_binary_paulis.values()]
    return self.grouped_paulis