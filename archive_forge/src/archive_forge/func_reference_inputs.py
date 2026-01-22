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
def reference_inputs(self, device, dtype, requires_grad=False, **kwargs):
    """
        Returns an iterable of SampleInputs.

        Distinct from sample_inputs() above because this returns an expanded set
        of inputs when reference_inputs_func is defined. If undefined this returns
        the sample inputs.
        """
    if self.reference_inputs_func is None:
        samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
        return TrackedInputIter(iter(samples), 'sample input')
    if kwargs.get('include_conjugated_inputs', False):
        raise NotImplementedError
    references = self.reference_inputs_func(self, device, dtype, requires_grad, **kwargs)
    return TrackedInputIter(iter(references), 'reference input')