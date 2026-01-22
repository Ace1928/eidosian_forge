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
def reduce_output(self, tensor: torch.Tensor, mask: torch.BoolTensor) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
    """
        Reduce transformer output at end of forward pass.

        :param tensor:
            encoded input
        :param mask:
            mask for encoded input

        :return (tensor, mask):
            returns the reduced tensor, and mask if appropriate
        """
    tensor *= self.output_scaling
    if self.reduction_type == 'first':
        return (tensor[:, 0, :], None)
    elif self.reduction_type == 'max':
        return (tensor.max(dim=1)[0], None)
    elif self.reduction_type == 'mean':
        divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
        output = tensor.sum(dim=1) / divisor
        return (output, None)
    elif self.reduction_type is None or 'none' in self.reduction_type:
        return (tensor, mask)
    else:
        raise ValueError("Can't handle --reduction-type {}".format(self.reduction_type))