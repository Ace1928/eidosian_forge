import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
def transform_wb_pesq_range(x: float) -> float:
    """The metric defined by ITU-T P.862 is often called 'PESQ score', which is defined
    for narrow-band signals and has a value range of [-0.5, 4.5] exactly. Here, we use the metric
    defined by ITU-T P.862.2, commonly known as 'wide-band PESQ' and will be referred to as "PESQ score".

    Args:
        x (float): Narrow-band PESQ score.

    Returns:
        (float): Wide-band PESQ score.
    """
    return 0.999 + (4.999 - 0.999) / (1 + math.exp(-1.3669 * x + 3.8224))