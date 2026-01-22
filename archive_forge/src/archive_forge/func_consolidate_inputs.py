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
@register_strategy('Consolidate Inputs')
def consolidate_inputs(cur_graph, cur_inps, granularity):
    old_len = len(cur_inps)
    cur_graph = _consolidate_placeholders(cur_graph, cur_inps)
    if len(cur_inps) > old_len and graph_fails(cur_graph, cur_inps):
        return ReproState(cur_graph, cur_inps)
    return None