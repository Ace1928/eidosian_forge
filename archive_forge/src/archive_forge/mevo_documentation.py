from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
Eval time forward that doesn't fuse the softmax and NLL Loss kernels.