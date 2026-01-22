import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Set, Tuple, Type, Union
import torch
from ..._cpp_lib import _built_with_cuda
from ..common import BaseOperator
from .attn_bias import (
@property
def scale_float(self) -> float:
    return self.query.shape[-1] ** (-0.5) if self.scale is None else self.scale