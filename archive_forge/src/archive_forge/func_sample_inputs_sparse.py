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
def sample_inputs_sparse(self, layout, device, dtype, requires_grad=False, **kwargs):
    """Returns an iterable of SampleInputs that contain inputs with a
        specified sparse layout.
        """
    layout_name = str(layout).split('.')[-1]
    sample_inputs_mth = getattr(self, 'sample_inputs_' + layout_name)

    def non_empty_sampler(op, generator):
        found_sample = False
        for sample in generator:
            found_sample = True
            yield sample
        if not found_sample:
            raise unittest.SkipTest('NO SAMPLES!')
    return non_empty_sampler(self, sample_inputs_mth(device, dtype, requires_grad=requires_grad, **kwargs))