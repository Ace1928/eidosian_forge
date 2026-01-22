import warnings
from typing import Optional, Tuple, Union
import torch
import torch.fx
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gptj import GPTJConfig
@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
def deparallelize(self):
    warnings.warn('Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.', FutureWarning)
    self.transformer.deparallelize()
    self.transformer = self.transformer.to('cpu')
    self.lm_head = self.lm_head.to('cpu')
    self.model_parallel = False
    torch.cuda.empty_cache()