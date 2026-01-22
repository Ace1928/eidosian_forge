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
def error_inputs_sparse(self, device, layout, **kwargs):
    """
        Returns an iterable of ErrorInputs that contain sparse sample
        inputs with a specified layout.
        """
    if not self.supports_sparse_layout(layout):
        raise unittest.SkipTest('unsupported sparse layout')
    return self.error_inputs_sparse_func(self, device, layout, **kwargs)