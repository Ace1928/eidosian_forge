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
def forward_layers(self, tensor: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """
        Apply transformer layers to input.

        :param tensor:
            embedded input
        :param mask:
            mask of input

        :return tensor:
            return embedding after applying transformer layers
        """
    if getattr(self.layers, 'is_model_parallel', False):
        tensor = self._apply_model_parallel(tensor, mask)
    else:
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)
    return tensor