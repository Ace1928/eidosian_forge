import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def in_base(self, key: _K) -> bool:
    """Checks if a key is in the base dictionary."""
    return key in self._base