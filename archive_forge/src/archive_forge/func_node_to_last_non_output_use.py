import collections
from typing import Any, Callable, Dict, Optional
import torch
import torch.utils._pytree as pytree
def node_to_last_non_output_use(self):
    last_non_output_use = collections.defaultdict(list)
    seen_uses = set()
    output_node = next(iter(reversed(self.module.graph.nodes)))
    for node in reversed(self.module.graph.nodes):
        if node.target == 'output':
            continue

        def add_use(inp):
            if inp in seen_uses:
                return
            seen_uses.add(inp)
            last_non_output_use[node].append(inp)
        pytree.tree_map_only(torch.fx.Node, add_use, (node.args, node.kwargs))
        if len(node.users) == 1 and output_node in node.users:
            last_non_output_use[node].append(node)
    return last_non_output_use