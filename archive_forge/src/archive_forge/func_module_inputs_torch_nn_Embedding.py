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
def module_inputs_torch_nn_Embedding(module_info, device, dtype, requires_grad, training, **kwargs):
    make_empty = partial(torch.empty, device=device, dtype=torch.long, requires_grad=False)
    return [ModuleInput(constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3), forward_input=FunctionInput(make_empty(2, 3).random_(4))), ModuleInput(constructor_input=FunctionInput(num_embeddings=4, embedding_dim=3), forward_input=FunctionInput(make_empty(1, 512).random_(4).expand(7, 512)), desc='discontiguous')]