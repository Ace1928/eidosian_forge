from contextlib import nullcontext
from typing import Any, ContextManager, Dict, Literal, Optional, Union
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from lightning_fabric.utilities.types import _PARAMETERS, Optimizable
def convert_module(self, module: Module) -> Module:
    """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
    return module