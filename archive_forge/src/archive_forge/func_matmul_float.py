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
def matmul_float(a: torch.Tensor, b: torch.Tensor, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    """
        Perform standard torch matrix multiplication, converting the result to the specified output data type.

        Args:
        a (torch.Tensor): The first matrix.
        b (torch.Tensor): The second matrix.
        output_dtype (Optional[torch.dtype]): The desired output data type.

        Returns:
        torch.Tensor: The result of the matrix multiplication.
        """
    return (a @ b).to(output_dtype)