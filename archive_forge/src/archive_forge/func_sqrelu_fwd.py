import math
import torch
import torch.nn as nn
import torch.nn.functional as F
@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return (r * r).to(dtype=x.dtype)