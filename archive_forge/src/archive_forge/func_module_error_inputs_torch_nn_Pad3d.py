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
def module_error_inputs_torch_nn_Pad3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    is_constant = kwargs.get('is_constant', False)
    return [ErrorModuleInput(ModuleInput(constructor_input=FunctionInput(1, 3) if is_constant else FunctionInput(3), forward_input=FunctionInput(make_input((2, 3)))), error_on=ModuleErrorEnum.FORWARD_ERROR, error_type=ValueError, error_regex='expected 4D or 5D input \\(got 2D input\\)')]