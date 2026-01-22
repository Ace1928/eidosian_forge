from typing import List
import torch
from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import tuple_sorted
def earliest_comm_descendant(node):
    for idx in range(len(comm_nodes)):
        if node in comm_ancestors[comm_nodes[idx]]:
            return idx
    return len(comm_nodes)