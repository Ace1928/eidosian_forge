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
def prepare_trace(self):
    self.profiler = prof.profile(use_cuda=ProfilerActivity.CUDA in self.activities, use_cpu=ProfilerActivity.CPU in self.activities, use_mtia=ProfilerActivity.MTIA in self.activities, use_device=None, record_shapes=self.record_shapes, with_flops=self.with_flops, profile_memory=self.profile_memory, with_stack=self.with_stack, with_modules=self.with_modules, use_kineto=True, experimental_config=self.experimental_config)
    self.profiler._prepare_trace()