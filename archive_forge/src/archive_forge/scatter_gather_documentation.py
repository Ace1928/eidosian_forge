import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, overload
from ._functions import Scatter, Gather
import warnings
Gather tensors from different GPUs on a specified device.

    Use 'cpu' for CPU to avoid a deprecation warning.
    