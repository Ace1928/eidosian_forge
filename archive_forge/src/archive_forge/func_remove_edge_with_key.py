import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def remove_edge_with_key(self, key):
    try:
        u, v, _ = self.edge_index[key]
    except KeyError as err:
        raise KeyError(f'Invalid edge key {key!r}') from err
    else:
        del self.edge_index[key]
        self._cls.remove_edge(u, v, key)