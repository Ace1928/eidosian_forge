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
def module_error_inputs_torch_nn_LSTMCell(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    samples = [ErrorModuleInput(ModuleInput(constructor_input=FunctionInput(10, 20), forward_input=FunctionInput(make_input(3, 11), (make_input(3, 20), make_input(3, 20)))), error_on=ModuleErrorEnum.FORWARD_ERROR, error_type=RuntimeError, error_regex='input has inconsistent input_size: got 11 expected 10'), ErrorModuleInput(ModuleInput(constructor_input=FunctionInput(10, 20), forward_input=FunctionInput(make_input(3, 10), (make_input(3, 21), make_input(3, 21)))), error_on=ModuleErrorEnum.FORWARD_ERROR, error_type=RuntimeError, error_regex='hidden0 has inconsistent hidden_size: got 21, expected 20'), ErrorModuleInput(ModuleInput(constructor_input=FunctionInput(10, 20), forward_input=FunctionInput(make_input(3, 10), (make_input(5, 20), make_input(5, 20)))), error_on=ModuleErrorEnum.FORWARD_ERROR, error_type=RuntimeError, error_regex="Input batch size 3 doesn't match hidden0 batch size 5"), ErrorModuleInput(ModuleInput(constructor_input=FunctionInput(10, 20), forward_input=FunctionInput(make_input(3, 10), (make_input(3, 1, 1, 20), make_input(3, 1, 1, 20)))), error_on=ModuleErrorEnum.FORWARD_ERROR, error_type=ValueError, error_regex='Expected hx\\[0\\] to be 1D or 2D, got 4D instead')]
    return samples