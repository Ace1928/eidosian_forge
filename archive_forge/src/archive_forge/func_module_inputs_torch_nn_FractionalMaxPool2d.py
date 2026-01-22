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
def module_inputs_torch_nn_FractionalMaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_random_samples():
        return torch.empty((1, 3, 2), dtype=torch.double, device=device).uniform_()
    return [ModuleInput(constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((1, 3, 5, 7))), desc='ratio'), ModuleInput(constructor_input=FunctionInput((2, 3), output_size=(4, 3), _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((1, 3, 7, 6))), desc='size'), ModuleInput(constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples(), return_indices=True), forward_input=FunctionInput(make_input((1, 3, 5, 7))), desc='ratio_return_indices'), ModuleInput(constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((3, 5, 7))), reference_fn=no_batch_dim_reference_fn, desc='ratio_no_batch_dim'), ModuleInput(constructor_input=FunctionInput((2, 3), output_size=(4, 3), _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((3, 7, 6))), reference_fn=no_batch_dim_reference_fn, desc='size_no_batch_dim')]