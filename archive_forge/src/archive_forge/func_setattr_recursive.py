import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import map_arg
from torch.fx.passes.utils import HolderModule, lift_subgraph_as_module
from .tools_common import NodeList
@compatibility(is_backward_compatible=False)
def setattr_recursive(obj, attr, value):
    if '.' not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split('.')
        setattr_recursive(getattr(obj, layer[0]), '.'.join(layer[1:]), value)