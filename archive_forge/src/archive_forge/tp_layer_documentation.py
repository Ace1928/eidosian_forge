import importlib
import warnings
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init
from peft.tuners.tuners_utils import BaseTunerLayer
from .layer import LoraLayer

    When the target layer parallel_linear is RowParallelLinear, in order to keep the input and output shapes
    consistent, we need to split the lora matrix A into rows, and the lora_B at this time should be a complete linear
    layer; In the same way, when the target layer is ColumnParallelLinear, we perform column segmentation on lora_B,
    while lora_A is still a complete linear layer.
    