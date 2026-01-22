from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
@property
def physical_qid_to_int(self) -> Dict['cirq.Qid', int]:
    """Mapping of physical qubits, that were part of the initial mapping, to unique integers."""
    return self._physical_qid_to_int