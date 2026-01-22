import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
def is_cuda_launch_kernel(e):
    return e.name == 'cudaLaunchKernel'