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
def conjugate_sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
    """Returns an iterable of SampleInputs but with the tensor input or first
        tensor in a sequence input conjugated.
        """
    samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
    conj_samples = list(samples)

    def conjugate(tensor):
        _requires_grad = tensor.requires_grad
        tensor = tensor.conj()
        return tensor.requires_grad_(_requires_grad)
    for i, sample in enumerate(samples):
        sample = conj_samples[i]
        if isinstance(sample.input, torch.Tensor):
            sample.input = conjugate(sample.input)
        else:
            sample.input[0] = conjugate(sample.input[0])
    return TrackedInputIter(iter(conj_samples), 'conjugate sample input')