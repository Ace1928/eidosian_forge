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
def try_granularity(failing_state, granularity, use_non_granular):
    print(f'Trying granularity {granularity}', file=sys.stderr)
    strategies = []
    num_nodes = len(failing_state.graph.nodes)
    num_outputs = len(get_outputs(failing_state.graph))
    if num_outputs > num_nodes // 2:
        strategies += [remove_outputs]
    if use_non_granular:
        strategies += [eliminate_dead_code, remove_unused_inputs, consolidate_inputs]
    strategies += [remove_suffix, delta_debugging]
    for strategy in strategies:
        new_state = strategy(failing_state, granularity)
        if new_state is not None:
            return new_state
    return None