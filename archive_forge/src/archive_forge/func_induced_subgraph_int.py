from typing import List, Dict, Sequence, TYPE_CHECKING
import networkx as nx
import numpy as np
@property
def induced_subgraph_int(self) -> nx.Graph:
    """Induced subgraph on physical qubit integers present in `self.logical_to_physical`."""
    return self._induced_subgraph_int