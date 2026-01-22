import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
def hdemucs_low(sources: List[str]) -> HDemucs:
    """Builds low nfft (1024) version of :class:`HDemucs`, suitable for sample rates around 8 kHz.

    Args:
        sources (List[str]): See :py:func:`HDemucs`.

    Returns:
        HDemucs:
            HDemucs model.
    """
    return HDemucs(sources=sources, nfft=1024, depth=5)