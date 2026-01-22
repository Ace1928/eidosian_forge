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
def sample_inputs_sparse_bsr(self, device, dtype, requires_grad=False, **kwargs):
    """Returns an iterable of SampleInputs that contain inputs with sparse
        bsr layout.
        """
    return self.sample_inputs_sparse_bsr_func(self, device, dtype, requires_grad, **kwargs)