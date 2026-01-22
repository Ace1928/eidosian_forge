import collections
from typing import Any, Callable, Dict, Optional
import torch
import torch.utils._pytree as pytree
def replace_node_with_constant(gm, node, constant):
    g = gm.graph
    if not hasattr(gm, '_frozen_param_count'):
        gm._frozen_param_count = 0
    i = gm._frozen_param_count
    while True:
        qualname = f'_frozen_param{i}'
        if not hasattr(gm, qualname):
            break
        i += 1
    gm._frozen_param_count = i + 1
    with g.inserting_before(node):
        new_input_node = g.create_node('get_attr', qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)