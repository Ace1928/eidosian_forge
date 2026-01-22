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
def padding3d_circular_ref(inp, pad):
    """input:
                [[[[[ 0.,  1.,  2.],
                    [ 3.,  4.,  5.]],
                   [[ 6.,  7.,  8.],
                    [ 9., 10., 11.]]]]]
            pad: (1, 2, 2, 1, 1, 2)
            output: [[[[[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]],

                       [[ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.]],

                       [[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]],

                       [[ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.],
                        [ 5.,  3.,  4.,  5.,  3.,  4.],
                        [ 2.,  0.,  1.,  2.,  0.,  1.]],

                       [[ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.],
                        [11.,  9., 10., 11.,  9., 10.],
                        [ 8.,  6.,  7.,  8.,  6.,  7.]]]]]
        """
    inp = torch.cat([inp[:, :, -pad[4]:], inp, inp[:, :, :pad[5]]], dim=2)
    inp = torch.cat([inp[:, :, :, -pad[2]:], inp, inp[:, :, :, :pad[3]]], dim=3)
    return torch.cat([inp[:, :, :, :, -pad[0]:], inp, inp[:, :, :, :, :pad[1]]], dim=4)