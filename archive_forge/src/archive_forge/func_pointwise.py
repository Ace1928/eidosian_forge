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
def pointwise(size_hints, triton_meta, tile_hint=None, filename=None, min_elem_per_thread=0, inductor_meta=None):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))
    hinted_configs = autotune_hints_to_configs(inductor_meta.get('autotune_hints', set()), size_hints, bs)
    triton_config_with_settings = functools.partial(triton_config, min_elem_per_thread=min_elem_per_thread)
    if len(size_hints) == 1:
        if disable_pointwise_autotuning() and (not (config.max_autotune or config.max_autotune_pointwise)):
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, bs)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
        else:
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, bs, num_elements_per_warp=256), triton_config_with_settings(size_hints, bs // 2, num_elements_per_warp=64), *hinted_configs], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
    if len(size_hints) == 2:
        if (disable_pointwise_autotuning() or tile_hint == TileHint.SQUARE) and (not (config.max_autotune or config.max_autotune_pointwise)):
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 32, 32)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
        return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 32, 32), triton_config_with_settings(size_hints, 64, 64), triton_config_with_settings(size_hints, 256, 16), triton_config_with_settings(size_hints, 16, 256), triton_config_with_settings(size_hints, bs, 1), triton_config_with_settings(size_hints, 1, bs), *hinted_configs], triton_meta=triton_meta, inductor_meta=inductor_meta, filename=filename, heuristic_type=HeuristicType.POINTWISE)
    if len(size_hints) == 3:
        if disable_pointwise_autotuning():
            return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 16, 16, 16)], triton_meta=triton_meta, inductor_meta=inductor_meta, heuristic_type=HeuristicType.POINTWISE, filename=filename)
        return cached_autotune(size_hints, [triton_config_with_settings(size_hints, 16, 16, 16), triton_config_with_settings(size_hints, 64, 8, 8), triton_config_with_settings(size_hints, 8, 64, 8), triton_config_with_settings(size_hints, 8, 8, 64), triton_config_with_settings(size_hints, bs, 1, 1), triton_config_with_settings(size_hints, 1, bs, 1), triton_config_with_settings(size_hints, 1, 1, bs), *hinted_configs], triton_meta=triton_meta, inductor_meta=inductor_meta, filename=filename, heuristic_type=HeuristicType.POINTWISE)
    raise NotImplementedError(f'size_hints: {size_hints}')