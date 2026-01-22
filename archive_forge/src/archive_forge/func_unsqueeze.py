from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def unsqueeze(self, dim: int) -> Rigid:
    """
        Analogous to torch.unsqueeze. The dimension is relative to the shared dimensions of the rotation/translation.

        Args:
            dim: A positive or negative dimension index.
        Returns:
            The unsqueezed transformation.
        """
    if dim >= len(self.shape):
        raise ValueError('Invalid dimension')
    rots = self._rots.unsqueeze(dim)
    trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)
    return Rigid(rots, trans)