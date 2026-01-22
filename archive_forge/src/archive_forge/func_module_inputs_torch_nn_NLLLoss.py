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
def module_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):

    def make_input(shape, device=device, dtype=dtype, requires_grad=requires_grad):
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=False).log_softmax(dim=1).requires_grad_(requires_grad)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    cases: List[Tuple[str, dict]] = [('', {}), ('reduction_sum', {'reduction': 'sum'}), ('reduction_none', {'reduction': 'none'}), ('ignore_index', {'ignore_index': 2}), ('weights', {'weight': make_weight(10).abs()}), ('weights_ignore_index', {'weight': make_weight(10).abs(), 'ignore_index': 2}), ('weights_ignore_index_neg', {'weight': make_weight(10).abs(), 'ignore_index': -1})]
    module_inputs = []
    for desc, constructor_kwargs in cases:

        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return nllloss_reference(i, t, **constructor_kwargs)
        module_inputs.append(ModuleInput(constructor_input=FunctionInput(**constructor_kwargs), forward_input=FunctionInput(make_input((15, 10)), torch.empty(15, device=device).uniform_().mul(10).floor().long()), desc=desc, reference_fn=reference_fn))
    return module_inputs