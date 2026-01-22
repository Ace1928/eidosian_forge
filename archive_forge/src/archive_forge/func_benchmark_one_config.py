import builtins
import copy
import functools
import hashlib
import inspect
import json
import logging
import math
import operator
import os
import os.path
import re
import threading
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import torch
import torch.autograd.profiler as autograd_profiler
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import dynamo_timed
from torch.utils._triton import has_triton, has_triton_package
from . import config
from .codecache import cache_dir, CudaKernelParamCache
from .coordinate_descent_tuner import CoordescTuner
from .ir import ReductionHint, TileHint
from .utils import (
def benchmark_one_config(config):
    with self.lock:
        _, launcher = self._precompile_config(config, None)
    config2launcher[config] = launcher
    out = self.bench(launcher, *cloned_args, **kwargs)
    log.debug('COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d', launcher.config, out, launcher.n_regs, launcher.n_spills, launcher.shared)
    return out