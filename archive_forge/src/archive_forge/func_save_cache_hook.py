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
def save_cache_hook(cfg, found_by_coordesc=False):
    with open(cache_filename, 'w') as fd:
        fd.write(json.dumps({**cfg.kwargs, 'num_warps': cfg.num_warps, 'num_stages': cfg.num_stages, 'configs_hash': configs_hash, 'found_by_coordesc': found_by_coordesc}))
    if log.isEnabledFor(logging.DEBUG):
        type_str = 'coordesc' if found_by_coordesc else 'heuristic'
        log.debug('Save %s tuning result to %s', type_str, cache_filename)