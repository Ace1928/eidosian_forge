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
def reorder_incremental_state(self, incremental_state: Dict[str, torch.Tensor], inds: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
        Reorder the input incremental-state tensors.
        """
    return {key: torch.index_select(val, 0, inds.to(val.device)).contiguous() for key, val in incremental_state.items()}