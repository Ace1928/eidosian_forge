import torch
import unittest
from copy import deepcopy
from enum import Enum
from functools import wraps, partial
from itertools import chain, product
import itertools
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_nn import nllloss_reference, get_reduction
from torch.testing._internal.common_utils import (
from types import ModuleType
from typing import List, Tuple, Type, Set, Dict
def module_inputs_torch_nn_GaussianNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    cases: List[Tuple[str, dict]] = [('', {}), ('reduction_sum', {'reduction': 'sum'}), ('reduction_mean', {'reduction': 'mean'}), ('reduction_none', {'reduction': 'none'})]
    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(ModuleInput(constructor_input=FunctionInput(**constructor_kwargs), forward_input=FunctionInput(make_input(3), make_target(3), make_input(1).abs()), desc=desc, reference_fn=no_batch_dim_reference_fn))
    return module_inputs