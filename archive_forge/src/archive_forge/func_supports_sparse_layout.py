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
def supports_sparse_layout(self, layout):
    """Return True if OpInfo supports the specified sparse layout."""
    layout_name = str(layout).split('.')[-1]
    layout_name = layout_name.replace('_coo', '')
    return getattr(self, f'supports_{layout_name}')