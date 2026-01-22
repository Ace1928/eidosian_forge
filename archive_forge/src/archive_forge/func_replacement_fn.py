import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
def replacement_fn(node):
    new_node = replacement_mapping[node]
    new_node.meta = node.meta.copy()
    return new_node