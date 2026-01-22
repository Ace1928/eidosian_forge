from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
def is_adjacent(self, lq1: int, lq2: int) -> bool:
    """Finds whether logical qubits `lq1` and `lq2` are adjacent on the device.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Returns:
            True, if physical qubits corresponding to `lq1` and `lq2` are adjacent on
            the device.
        """
    return self.dist_on_device(lq1, lq2) == 1