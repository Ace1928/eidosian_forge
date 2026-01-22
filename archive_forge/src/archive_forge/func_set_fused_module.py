from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_fused_module(self, fused_module: Type[torch.nn.Module]) -> BackendPatternConfig:
    """
        Set the module that represents the fused implementation for this pattern.
        """
    self.fused_module = fused_module
    return self