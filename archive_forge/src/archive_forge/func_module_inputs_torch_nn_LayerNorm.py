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
def module_inputs_torch_nn_LayerNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [ModuleInput(constructor_input=FunctionInput([5], 0.001), forward_input=FunctionInput(make_input((4, 5, 5))), desc='1d_elementwise_affine'), ModuleInput(constructor_input=FunctionInput([5], 0.001), forward_input=FunctionInput(make_input((128, 5, 5))), desc='1d_elementwise_affine_large_batch'), ModuleInput(constructor_input=FunctionInput([5], 0.001, False), forward_input=FunctionInput(make_input((4, 5, 5))), desc='1d_no_elementwise_affine'), ModuleInput(constructor_input=FunctionInput([2, 2, 5], 0.001), forward_input=FunctionInput(make_input((4, 2, 2, 5))), desc='3d_elementwise_affine'), ModuleInput(constructor_input=FunctionInput([2, 2, 5], 0.001, False), forward_input=FunctionInput(make_input((4, 2, 2, 5))), desc='3d_no_elementwise_affine'), ModuleInput(constructor_input=FunctionInput([5], 0.001), forward_input=FunctionInput(make_input((0, 5))), desc='1d_empty_elementwise_affine'), ModuleInput(constructor_input=FunctionInput([2, 2, 5], 0.001, elementwise_affine=True, bias=False), forward_input=FunctionInput(make_input((4, 2, 2, 5))), desc='3d_elementwise_affine_no_bias')]