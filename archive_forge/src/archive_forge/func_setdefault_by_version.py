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
def setdefault_by_version(self, key: TensorKey, version: int, category: Category) -> None:
    self._values[key.id].by_version.setdefault((key, version), category)