from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_pattern(self, pattern: Pattern) -> BackendPatternConfig:
    """
        Set the pattern to configure.

        The pattern can be a float module, functional operator, pytorch operator, or a tuple
        combination of the above. Tuple patterns are treated as sequential patterns, and
        currently only tuples of 2 or 3 elements are supported.
        """
    if self._pattern_complex_format is not None:
        raise ValueError("Only one of 'pattern' or 'pattern_complex_format' can be set")
    self.pattern = pattern
    return self