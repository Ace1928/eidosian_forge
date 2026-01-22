from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def get_trans(self) -> torch.Tensor:
    """
        Getter for the translation.

        Returns:
            The stored translation
        """
    return self._trans