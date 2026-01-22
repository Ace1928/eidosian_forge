from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
@property
def int_to_logical_qid(self) -> List['cirq.Qid']:
    """Inverse mapping of unique integers to corresponding physical qubits.

        `self.logical_qid_to_int[self.int_to_logical_qid[i]] == i` for each i.
        """
    return self._int_to_logical_qid