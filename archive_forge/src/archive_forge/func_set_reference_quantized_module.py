from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_reference_quantized_module(self, reference_quantized_module: Type[torch.nn.Module]) -> BackendPatternConfig:
    """
        Set the module that represents the reference quantized implementation for
        this pattern's root module.

        For more detail, see :func:`~torch.ao.quantization.backend_config.BackendPatternConfig.set_root_module`.
        """
    self.reference_quantized_module = reference_quantized_module
    return self