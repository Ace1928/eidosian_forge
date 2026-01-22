import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def set_tensor_metadata(tensor, metadata):
    assert isinstance(metadata, dict)
    assert isinstance(tensor, torch.Tensor)
    torch._C._set_tensor_metadata(tensor, metadata)