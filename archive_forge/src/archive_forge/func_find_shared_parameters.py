from typing import Dict, List, Optional
from torch import nn
def find_shared_parameters(module: nn.Module) -> List[str]:
    """Returns a list of names of shared parameters set in the module."""
    return _find_shared_parameters(module)