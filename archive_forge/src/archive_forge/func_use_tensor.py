from typing import List, Optional, Tuple
import torch
from torch import Tensor
from ..copy import Context as CopyContext
from ..copy import Copy
from ..phony import get_phony
from ..stream import AbstractStream, get_device
def use_tensor(self) -> Optional[Tensor]:
    """Retrieves the underlying tensor and decreases the tensor  life. When
        the life becomes 0, it the tensor will be removed.
        """
    self.check_tensor_life()
    tensor = self.tensor
    self.tensor_life -= 1
    if self.tensor_life <= 0:
        self.tensor = None
    return tensor