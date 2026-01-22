import collections
import warnings
from functools import partial, wraps
from typing import Sequence
import numpy as np
import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict
def str_format_dynamic_dtype(op):
    fmt_str = f'\n        OpInfo({op.name},\n               dtypes={dtypes_dispatch_hint(op.dtypes).dispatch_fn_str},\n               dtypesIfCUDA={dtypes_dispatch_hint(op.dtypesIfCUDA).dispatch_fn_str},\n        )\n        '
    return fmt_str