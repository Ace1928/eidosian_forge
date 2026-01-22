from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
@property
def physical_to_logical(self) -> np.ndarray:
    """The mapping of physical qubits integers to logical qubits integers.

        Let `pq: cirq.Qid` be a physical qubit. Then the corresponding logical qubit that it
        maps to can be obtained by:
        `self.int_to_logical_qid[self.physical_to_logical[self.physical_qid_to_int[pq]]]`
        """
    return self._physical_to_logical