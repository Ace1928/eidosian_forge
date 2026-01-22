from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
def dist_on_device(self, lq1: int, lq2: int) -> int:
    """Finds distance between logical qubits 'lq1' and 'lq2' on the device.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Returns:
            The shortest path distance.
        """
    return self._distances[self.logical_to_physical[lq1]][self.logical_to_physical[lq2]]