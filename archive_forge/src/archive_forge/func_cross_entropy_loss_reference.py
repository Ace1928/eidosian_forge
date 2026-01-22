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
def cross_entropy_loss_reference(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    if input.shape == target.shape:
        return cross_entropy_loss_prob_target_reference(input, target, weight=weight, reduction=reduction, label_smoothing=label_smoothing)
    else:
        return cross_entropy_loss_indices_target_reference(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index, label_smoothing=label_smoothing)