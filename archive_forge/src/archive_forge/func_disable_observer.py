import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
@torch.jit.export
def disable_observer(self):
    self.enable_observer(False)