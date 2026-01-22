import logging
import operator
from collections import defaultdict
from typing import Set
import torch
from torch.fx import GraphModule
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn import Module
from torch.utils._pytree import tree_map
from .common import aot_autograd
from .registry import register_backend
def find_input_mutations(g):

    def meta_fk(meta):
        return meta['val'] if 'val' in meta else meta['fake_result']
    inputs = defaultdict(set)
    input_idx = 0
    mutated_inputs = set()
    for n in g.nodes:
        if n.op == 'placeholder':
            inputs[StorageWeakRef(meta_fk(n.meta)._typed_storage())].add(input_idx)
            input_idx += 1
        elif n.op == 'call_function':
            if n.target is operator.getitem:
                continue
            schema = n.target._schema
            for i, arg in enumerate(schema.arguments):
                if i < len(n.args):
                    argument = n.args[i]
                else:
                    if arg.name not in n.kwargs:
                        continue
                    argument = n.kwargs[arg.name]
                mut_arg = False
                if arg.alias_info:
                    if arg.alias_info.is_write:
                        mut_arg = True
                if mut_arg:
                    mutated_inputs |= inputs[StorageWeakRef(meta_fk(argument.meta)._typed_storage())]
    return mutated_inputs