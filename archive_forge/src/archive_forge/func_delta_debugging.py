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
@register_strategy('Delta Debugging')
def delta_debugging(cur_graph: fx.Graph, cur_inps, granularity):
    num_nodes = len(cur_graph.nodes)
    for start_range in range(0, num_nodes, granularity):
        is_removing = False
        new_graph = deepcopy_fx_graph(cur_graph)
        new_inps = cur_inps[:]
        end_range = min(num_nodes, start_range + granularity)
        for idx in range(start_range, end_range):
            new_node = list(new_graph.nodes)[idx]
            if _convert_node_to_placeholder(new_graph, new_node, new_inps):
                is_removing = True
        if not is_removing:
            continue
        new_graph.eliminate_dead_code()
        new_graph = _consolidate_placeholders(new_graph, new_inps)
        new_state = remove_unused_inputs_unchecked(ReproState(new_graph, new_inps))
        if new_state is None:
            new_state = ReproState(new_graph, new_inps)
        if graph_fails(new_state.graph, new_state.inps):
            return ReproState(new_state.graph, new_state.inps)
    return None