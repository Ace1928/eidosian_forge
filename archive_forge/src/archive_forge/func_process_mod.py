import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
def process_mod(mod_name, depth):
    nonlocal is_global_subsumed
    total_flops = sum(self.flop_counts[mod_name].values())
    is_global_subsumed |= total_flops >= global_flops
    padding = ' ' * depth
    values = []
    values.append([padding + mod_name, convert_num_with_suffix(total_flops, global_suffix), convert_to_percent_str(total_flops, global_flops)])
    for k, v in self.flop_counts[mod_name].items():
        values.append([padding + ' - ' + str(k), convert_num_with_suffix(v, global_suffix), convert_to_percent_str(v, global_flops)])
    return values