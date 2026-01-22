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
def gradcheck_wrapper_masked_pointwise_operation(op, input, *args, **kwargs):
    """Gradcheck wrapper for masked pointwise operations. Assumes that the result
    will be masked iff both tensors are masked at a specific index

    When mask is specified, replaces masked-out elements with zeros.

    Use for operations that produce non-finite masked-out elements,
    for instance, for minimum and maximum reductions.
    """
    output = op(input, *args, **kwargs)
    input_mask = kwargs.get('input_mask')
    other_mask = kwargs.get('other_mask')
    if input_mask is not None and other_mask is not None:
        combined_mask = torch.logical_and(input_mask, other_mask)
        new_kwargs = dict(mask=combined_mask, **kwargs)
        output_mask = torch.masked._input_mask(input, *args, **new_kwargs)
        output = torch.where(output_mask, output, output.new_zeros([]))
    return output