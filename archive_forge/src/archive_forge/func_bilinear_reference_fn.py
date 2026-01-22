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
def bilinear_reference_fn(m, p, x1, x2, bias=True):
    result = torch.einsum('bn,anm,bm->ba', x1, p[0], x2)
    if bias:
        if x1.shape[0] == 1:
            result = result.view(-1) + p[1]
        else:
            result = result + p[1].view(1, -1).expand(x1.shape[0], p[0].shape[0])
    return result