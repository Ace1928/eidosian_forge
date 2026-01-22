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
def module_inputs_torch_nn_RNN_GRU(module_info, device, dtype, requires_grad, training, with_packed_sequence=False, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    is_rnn = kwargs['is_rnn']
    nonlinearity = ('relu', 'tanh')
    bias = (False, True)
    batch_first = (False, True)
    bidirectional = (False, True)
    samples = []
    if is_rnn:
        prod_gen = product(nonlinearity, bias, batch_first, bidirectional)
    else:
        prod_gen = product(bias, batch_first, bidirectional)
    for args in prod_gen:
        if is_rnn:
            nl, b, b_f, bidir = args
        else:
            b, b_f, bidir = args
        cons_args = {'input_size': 2, 'hidden_size': 2, 'num_layers': 2, 'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        cons_args_hidden = {'input_size': 2, 'hidden_size': 3, 'num_layers': 2, 'batch_first': b_f, 'bias': b, 'bidirectional': bidir}
        if is_rnn:
            cons_args['nonlinearity'] = nl
            cons_args_hidden['nonlinearity'] = nl
        samples.append(ModuleInput(constructor_input=FunctionInput(**cons_args), forward_input=FunctionInput(make_input((3, 2))), reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f)))
        samples.append(ModuleInput(constructor_input=FunctionInput(**cons_args_hidden), forward_input=FunctionInput(make_input((3, 2)), make_input((4 if bidir else 2, 3))), reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f)))
        if with_packed_sequence:
            samples.append(ModuleInput(constructor_input=FunctionInput(**cons_args), forward_input=FunctionInput(make_packed_sequence(make_input((5, 2, 2)), torch.tensor([5, 3]))), reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f)))
            samples.append(ModuleInput(constructor_input=FunctionInput(**cons_args), forward_input=FunctionInput(make_packed_sequence(make_input((5, 5, 2)), torch.tensor([5, 3, 3, 2, 2]))), reference_fn=partial(no_batch_dim_reference_rnn_gru, batch_first=b_f)))
    return samples