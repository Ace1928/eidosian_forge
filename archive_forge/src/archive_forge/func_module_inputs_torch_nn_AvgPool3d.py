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
def module_inputs_torch_nn_AvgPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return [ModuleInput(constructor_input=FunctionInput((2, 2, 2)), forward_input=FunctionInput(make_input((3, 4, 4, 4))), desc='no_batch_dim', reference_fn=no_batch_dim_reference_fn), ModuleInput(constructor_input=FunctionInput((2, 2, 2)), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4)))), ModuleInput(constructor_input=FunctionInput(2, (2, 2, 2)), forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))), desc='stride'), ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1)), forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))), desc='stride_pad'), ModuleInput(constructor_input=FunctionInput(4, 2, (1, 2, 1)), forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))), desc='stride_pad_gpu_fixedkw_output'), ModuleInput(constructor_input=FunctionInput((2, 4, 8), 1, (1, 1, 2)), forward_input=FunctionInput(make_input((2, 3, 2, 4, 8))), desc='stride_pad_gpu_general_output'), ModuleInput(constructor_input=FunctionInput(3, 1, 0), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='stride1_pad0_gpu_input'), ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1)), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='stride_pad_gpu_input_nooverlap'), ModuleInput(constructor_input=FunctionInput((2, 2, 2), divisor_override=1), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='divisor'), ModuleInput(constructor_input=FunctionInput(2, (2, 2, 2), divisor_override=1), forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))), desc='divisor_stride'), ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1), divisor_override=1), forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))), desc='divisor_stride_pad'), ModuleInput(constructor_input=FunctionInput(4, 2, (1, 2, 1), divisor_override=1), forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))), desc='divisor_stride_pad_gpu_fixedkw_output'), ModuleInput(constructor_input=FunctionInput((2, 4, 8), 1, (1, 1, 2), divisor_override=1), forward_input=FunctionInput(make_input((2, 3, 2, 4, 8))), desc='divisor_stride_pad_gpu_general_output'), ModuleInput(constructor_input=FunctionInput(3, 1, 0, divisor_override=1), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='divisor_stride1_pad0_gpu_input'), ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1), divisor_override=1), forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))), desc='divisor_stride_pad_gpu_input_nooverlap')]