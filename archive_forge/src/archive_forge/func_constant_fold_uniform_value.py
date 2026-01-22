import logging
import typing
from collections import Counter
from typing import Dict, Set
import torch
import torch._guards
from torch._inductor.constant_folding import ConstantFolder
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from ..pattern_matcher import (
from .replace_random import replace_random_passes
@torch.utils._python_dispatch._disable_current_modes()
def constant_fold_uniform_value(gm: torch.fx.GraphModule):
    """Runs constant folding and replaces constants which can be constructed with a single `full` call. Calls into remove_no_ops."""
    aten = torch.ops.aten
    cf = UniformValueConstantFolder(gm)
    cf.run()
    node_replacements = cf.node_replacements
    graph = gm.graph
    zeros = set()
    ones = set()
    constant_data_ptr_count: typing.Counter[StorageWeakRef] = Counter()
    for node in cf.node_replacements:
        constant_data_ptr_count[cf.constant_data_ptrs[node]] += 1
    for node, value in node_replacements.items():
        fake_tensor = node.meta['val']
        if not fake_tensor.is_contiguous(memory_format=torch.contiguous_format):
            continue
        if constant_data_ptr_count[cf.constant_data_ptrs[node]] > 1:
            continue
        with graph.inserting_after(node):
            if node.op == 'call_function' and node.target == aten.full.default and (len(node.args) == 2):
                value = node.args[1]
            new_node = graph.call_function(aten.full.default, args=(list(fake_tensor.shape), value), kwargs={'dtype': fake_tensor.dtype, 'layout': torch.strided, 'device': fake_tensor.device, 'pin_memory': False})
            new_node.meta.update(node.meta)
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            if value == 0:
                zeros.add(new_node)
            elif value == 1:
                ones.add(new_node)
    remove_no_ops(gm, zeros, ones)
    remove_redundant_views(gm)