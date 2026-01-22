import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
def create_position_codes(n_pos, dim, out):
    """
    Create positional codes and store them in ``out``.
    """
    position_enc = np.array([[pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False