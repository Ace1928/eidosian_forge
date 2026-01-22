from abc import abstractmethod
import tempfile
import unittest
from copy import deepcopy
from functools import reduce, partial, wraps
from itertools import product
from operator import mul
from math import pi
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, \
from torch.testing._internal.common_cuda import TEST_CUDA, SM90OrLater
from torch.autograd.gradcheck import _get_numerical_jacobian, _iter_tensors
from torch.autograd import Variable
from torch.types import _TensorOrTensors
import torch.backends.cudnn
from typing import Dict, Callable, Tuple, List, Sequence, Union, Any
def nlllossNd_no_reduce_ignore_index_test():
    t = Variable(torch.rand(2, 5, 5, 2, 2).mul(3).floor().long())
    kwargs: Dict[str, Union[int, str]] = {'ignore_index': 1, 'reduction': 'none'}
    return dict(fullname='NLLLossNd_no_reduce_ignore_index', constructor=wrap_functional(lambda i: F.nll_loss(i, t.type_as(i).long(), ignore_index=int(kwargs['ignore_index']), reduction=str(kwargs['reduction']))), cpp_function_call='F::nll_loss(\n            i, t.to(i.options()).to(torch::kLong), F::NLLLossFuncOptions().ignore_index(1).reduction(torch::kNone))', input_fn=lambda: torch.rand(2, 3, 5, 5, 2, 2).log(), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['NLLLossNd'](i, t.type_as(i).long(), **kwargs), pickle=False, default_dtype=torch.double)