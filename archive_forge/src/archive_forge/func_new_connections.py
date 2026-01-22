from __future__ import annotations
from typing import Any
import copy
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap
def new_connections(self, idx: int) -> list[set[int]]:
    """
        Returns the new connections obtained after applying the SWAP layer specified by idx, i.e.
        a list of qubit pairs that are adjacent to one another after idx steps of the SWAP strategy.

        Args:
            idx: The index of the SWAP layer. 1 refers to the first SWAP layer whereas an ``idx``
                of 0 will return the connections present in the original coupling map.

        Returns:
            A list of edges representing the new qubit connections.
        """
    connections = []
    for i in range(self._num_vertices):
        for j in range(i):
            if self.distance_matrix[i, j] == idx:
                connections.append({i, j})
    return connections