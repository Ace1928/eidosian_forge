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
def multilabelsoftmarginloss_weights_no_reduce_test():
    t = torch.rand(5, 10).mul(2).floor()
    weights = torch.rand(10)
    return dict(fullname='MultiLabelSoftMarginLoss_weights_no_reduce', constructor=wrap_functional(lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i), weight=weights.type_as(i), reduction='none')), cpp_function_call='F::multilabel_soft_margin_loss(\n            i, t.to(i.options()),\n            F::MultilabelSoftMarginLossFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 10), cpp_var_map={'i': '_get_input()', 't': t, 'weights': weights}, reference_fn=lambda i, *_: (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * weights).sum(dim=1) / i.size(1), check_sum_reduction=True, check_gradgrad=False, pickle=False, default_dtype=torch.double)