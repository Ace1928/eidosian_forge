from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def get_module_to_qat_module(backend_config: BackendConfig) -> Dict[Pattern, Type[torch.nn.Module]]:
    module_to_qat_module: Dict[Pattern, Type[torch.nn.Module]] = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        if config.qat_module is not None:
            module_to_qat_module[pattern] = config.qat_module
    return module_to_qat_module