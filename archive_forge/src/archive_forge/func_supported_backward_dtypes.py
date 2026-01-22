import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def supported_backward_dtypes(self, device_type):
    if not self.supports_autograd:
        return set()
    device_type = torch.device(device_type).type
    backward_dtypes = None
    if device_type == 'cuda':
        backward_dtypes = self.backward_dtypesIfROCM if TEST_WITH_ROCM else self.backward_dtypesIfCUDA
    else:
        backward_dtypes = self.backward_dtypes
    allowed_backward_dtypes = floating_and_complex_types_and(torch.bfloat16, torch.float16, torch.complex32)
    return set(allowed_backward_dtypes).intersection(backward_dtypes)