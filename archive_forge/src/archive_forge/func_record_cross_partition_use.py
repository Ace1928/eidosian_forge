import inspect
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
from collections import OrderedDict
import logging
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
def record_cross_partition_use(def_node: Node, use_node: Optional[Node]):
    from torch.fx.experimental.symbolic_shapes import free_symbols
    defined = getattr(def_node, '_fx_partition', None)
    used = getattr(use_node, '_fx_partition', None)
    if defined != used:
        if defined is not None:
            def_partition = partitions[defined]
            def_partition.outputs.setdefault(def_node.name)
            if used is not None:
                def_partition.dependents.setdefault(used)
        if used is not None:
            use_partition = partitions[used]
            use_partition.inputs.setdefault(def_node.name)
            if (def_val := def_node.meta.get('example_value')) is not None:
                for s in sorted(free_symbols(def_val)):
                    use_partition.inputs.setdefault(symbol_to_node[s].name)
            if defined is not None:
                use_partition.dependencies.setdefault(defined)