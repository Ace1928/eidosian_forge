import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List
import numpy as np
from numpy import inf
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo.core import (
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo
def sample_kwargs_vector_norm(t, **kwargs):

    def ords():
        has_id = (6, 4, 2, 1, 0, 0.9)
        no_id = (inf, -2.1, -inf)
        if t.numel() == 0:
            dim = kwargs.get('dim')
            if dim is None:
                return has_id
            if not isinstance(dim, Iterable):
                dim = (dim,)
            for d in dim:
                if t.size(d) == 0:
                    return has_id
        return has_id + no_id
    return (((), dict(ord=o)) for o in ords())