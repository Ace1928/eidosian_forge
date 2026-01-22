from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def sample_inputs_flash_attention_forward(op_info, device, dtype, requires_grad, **kwargs):
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    batch, num_heads, head_dim = (4, 4, 8)
    seq_q = 11
    seq_kv = 32
    dim_4_q_shape = (batch, num_heads, seq_q, head_dim)
    dim_4_kv_shape = (batch, num_heads, seq_kv, head_dim)
    qkv_shapes = [(dim_4_q_shape, dim_4_kv_shape)]
    samples = []
    scales = [None, 1.0]
    for qkv_shape, is_causal, dropout_p, scale in product(qkv_shapes, [True, False], [0.0, 0.5], scales):
        shape_q, shape_kv = qkv_shape
        samples.append(SampleInput(make(shape_q).transpose(1, 2), make(shape_kv).transpose(1, 2), make(shape_kv).transpose(1, 2), cum_seq_q=None, cum_seq_k=None, max_q=seq_q, max_k=seq_kv, dropout_p=dropout_p, is_causal=is_causal, return_debug_mask=False, scale=scale))
    yield from samples