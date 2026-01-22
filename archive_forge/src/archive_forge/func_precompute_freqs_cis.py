import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0) -> torch.Tensor:
    """
    Precomputes the frequency cosine and sine values for rotary embeddings.

    This function calculates the cosine and sine values used in rotary position embeddings, which are used to add positional information to the embeddings.

    Parameters:
        dim (int): The dimensionality of the embedding.
        end (int): The sequence length or the number of positions.
        theta (float): A scaling factor for the frequencies, defaulting to 10000.0.

    Returns:
        torch.Tensor: A tensor of precomputed cosine and sine values for rotary embeddings.
    """
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis