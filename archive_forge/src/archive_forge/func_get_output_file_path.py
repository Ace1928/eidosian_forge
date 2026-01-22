import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
def get_output_file_path(self) -> str:
    """
        Returns the output file name.
        """
    if self.is_registered:
        return self._output_file_path
    else:
        raise RuntimeError('A callback to the ET profiler needs to be registered first before getting the output file path')