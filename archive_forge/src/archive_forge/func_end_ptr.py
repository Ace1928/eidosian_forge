from __future__ import annotations
import dataclasses
from . import torch_wrapper
@property
def end_ptr(self) -> int:
    return self.ptr + self.size