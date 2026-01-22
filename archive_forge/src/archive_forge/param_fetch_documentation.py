from torch.fx.graph_module import GraphModule
from typing import Any, Callable, Dict, List, Tuple, Type
import torch
import torch.nn as nn
from torch.fx._compatibility import compatibility
Recursively traverse all `fx_module` nodes and fetch the module's attributes if the node is a leaf module.
    