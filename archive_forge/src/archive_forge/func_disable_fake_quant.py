import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
@torch.jit.export
def disable_fake_quant(self):
    self.enable_fake_quant(False)