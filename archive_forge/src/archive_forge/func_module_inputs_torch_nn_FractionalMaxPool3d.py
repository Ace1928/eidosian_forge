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
def module_inputs_torch_nn_FractionalMaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_random_samples():
        return torch.empty((2, 4, 3), dtype=torch.double, device=device).uniform_()
    return [ModuleInput(constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((2, 4, 5, 5, 5))), desc='ratio'), ModuleInput(constructor_input=FunctionInput((2, 2, 2), output_size=(4, 4, 4), _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((2, 4, 7, 7, 7))), desc='size'), ModuleInput(constructor_input=FunctionInput((4, 2, 3), output_size=(10, 3, 2), _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((2, 4, 16, 7, 5))), desc='asymsize'), ModuleInput(constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples(), return_indices=True), forward_input=FunctionInput(make_input((2, 4, 5, 5, 5))), desc='ratio_return_indices'), ModuleInput(constructor_input=FunctionInput(2, output_ratio=0.5, _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((4, 5, 5, 5))), reference_fn=no_batch_dim_reference_fn, desc='ratio_no_batch_dim'), ModuleInput(constructor_input=FunctionInput((2, 2, 2), output_size=(4, 4, 4), _random_samples=make_random_samples()), forward_input=FunctionInput(make_input((4, 7, 7, 7))), reference_fn=no_batch_dim_reference_fn, desc='size_no_batch_dim')]