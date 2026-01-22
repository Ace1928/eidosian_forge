from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
def check_tensor_life(self) -> None:
    if self.tensor_life <= 0:
        raise RuntimeError('tensor in portal has been removed')