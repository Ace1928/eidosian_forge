import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os
def remove_unused_inputs_unchecked(cur_state: ReproState):
    cur_graph = cur_state.graph
    cur_inps = cur_state.inps
    ph_nodes = get_placeholders(cur_graph)
    assert len(ph_nodes) == len(cur_inps)
    new_inps = []
    for idx in range(len(ph_nodes)):
        if len(ph_nodes[idx].users) == 0:
            cur_graph.erase_node(ph_nodes[idx])
        else:
            new_inps.append(cur_inps[idx])
    if len(new_inps) < len(cur_inps):
        return ReproState(cur_graph, new_inps)
    return None