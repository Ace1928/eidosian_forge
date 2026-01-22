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
def multilabelmarginloss_reference(input, target, reduction='mean'):
    input_dim = input.dim()
    if input.dim() < 2:
        assert target.dim() < 2
        input = input.unsqueeze(0) if input.dim() == 1 else input.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0) if target.dim() == 1 else target.unsqueeze(0).unsqueeze(0)
    n = input.size(0)
    dim = input.size(1)
    output = input.new(n).zero_()
    for i in range(0, n):
        output[i] = _multilabelmarginloss_reference(input[i], target[i])
    if reduction == 'mean':
        return output.mean() / dim
    elif reduction == 'sum':
        return output.sum() / dim
    elif input_dim < 2:
        return output.squeeze() / dim
    else:
        return output / dim