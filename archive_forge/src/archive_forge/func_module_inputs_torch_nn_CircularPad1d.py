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
def module_inputs_torch_nn_CircularPad1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def padding1d_circular_ref(inp, pad):
        """ input:
                [[[0., 1., 2.],
                  [3., 4., 5.]]]
                pad: (1, 2)
                output:
                    [[[2., 0., 1., 2., 0., 1.],
                      [5., 3., 4., 5., 3., 4.]]]
            """
        return torch.cat([inp[:, :, -pad[0]:], inp, inp[:, :, :pad[1]]], dim=2)
    return [ModuleInput(constructor_input=FunctionInput(1), forward_input=FunctionInput(make_input((3, 4))), reference_fn=no_batch_dim_reference_fn), ModuleInput(constructor_input=FunctionInput((1, 2)), forward_input=FunctionInput(make_input((1, 2, 3))), reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding)), ModuleInput(constructor_input=FunctionInput((3, 1)), forward_input=FunctionInput(make_input((1, 2, 3))), reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding)), ModuleInput(constructor_input=FunctionInput((3, 3)), forward_input=FunctionInput(make_input((1, 2, 3))), reference_fn=lambda m, p, i: padding1d_circular_ref(i, m.padding))]