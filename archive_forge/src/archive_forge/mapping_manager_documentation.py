from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
Find the shortest path between two logical qubits on the device, given their mapping.

        Args:
            lq1: integer corresponding to the first logical qubit.
            lq2: integer corresponding to the second logical qubit.

        Returns:
            A sequence of logical qubit integers on the shortest path from `lq1` to `lq2`.
        