import re
from typing import Callable, Dict, Optional, Set, Union
import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module
def mod_partition(node: torch.fx.Node):
    return 0 if node in const_nodes else 1