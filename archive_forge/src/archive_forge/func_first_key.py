import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def first_key(i, vv):
    keys = G[nodes[i]][vv].keys()
    keys = list(keys)
    return keys[0]