from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
def handle_pop(name: str) -> Optional[Tensor]:
    if name not in self.poppable_names:
        raise RuntimeError(f"'{name}' has not been declared as poppable")
    return poppable_tensors.pop(name)