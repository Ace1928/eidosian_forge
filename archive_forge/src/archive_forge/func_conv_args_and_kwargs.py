import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_args_and_kwargs(kwarg_names, expanded_args_and_kwargs):
    args = expanded_args_and_kwargs[:len(expanded_args_and_kwargs) - len(kwarg_names)]
    kwargs = expanded_args_and_kwargs[len(expanded_args_and_kwargs) - len(kwarg_names):]
    kwargs = dict(zip(kwarg_names, kwargs))
    return conv_normalizer(*args, **kwargs)