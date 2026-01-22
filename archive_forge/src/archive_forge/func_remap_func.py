import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import map_arg
from torch.fx.passes.utils import HolderModule, lift_subgraph_as_module
from .tools_common import NodeList
def remap_func(x):
    if x.op == 'get_attr':
        if x not in comp.getattr_maps:
            comp.getattr_maps[x] = comp.graph.get_attr(x.target, type_expr=x.type)
        return comp.getattr_maps[x]
    if x.op != 'placeholder' and node_to_component[x] == comp:
        return node_remapping[x]
    if x not in comp.orig_inputs:
        comp.orig_inputs.append(x)
        placeholder = comp.graph.placeholder(x.name, type_expr=x.type)
        placeholder.meta = copy.copy(x.meta)
        comp.input_placeholders.append(placeholder)
        used_in_main[x] = None
    return comp.input_placeholders[comp.orig_inputs.index(x)]