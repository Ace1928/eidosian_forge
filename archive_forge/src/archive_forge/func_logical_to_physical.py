from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
@property
def logical_to_physical(self) -> np.ndarray:
    """The mapping of logical qubit integers to physical qubit integers.

        Let `lq: cirq.Qid` be a logical qubit. Then the corresponding physical qubit that it
        maps to can be obtained by:
        `self.int_to_physical_qid[self.logical_to_physical[self.logical_qid_to_int[lq]]]`
        """
    return self._logical_to_physical