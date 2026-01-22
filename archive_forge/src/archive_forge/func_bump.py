import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def bump(self, key: TensorKey) -> None:
    prior_version = self._active_version.get(key, None)
    assert prior_version is not None
    self._active_version[key] = prior_version + 1