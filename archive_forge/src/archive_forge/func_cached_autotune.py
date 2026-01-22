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
def cached_autotune(size_hints: Optional[List[int]], configs: List[Config], triton_meta, heuristic_type, filename=None, inductor_meta=None):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    save_cache_hook: Optional[Callable[[Any, Any], Any]]
    inductor_meta = {} if inductor_meta is None else inductor_meta
    if filename is not None and (len(configs) > 1 or config.coordinate_descent_tuning):
        cache_filename = os.path.splitext(filename)[0] + '.best_config'
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg, found_by_coordesc=False):
            with open(cache_filename, 'w') as fd:
                fd.write(json.dumps({**cfg.kwargs, 'num_warps': cfg.num_warps, 'num_stages': cfg.num_stages, 'configs_hash': configs_hash, 'found_by_coordesc': found_by_coordesc}))
            if log.isEnabledFor(logging.DEBUG):
                type_str = 'coordesc' if found_by_coordesc else 'heuristic'
                log.debug('Save %s tuning result to %s', type_str, cache_filename)
    else:
        save_cache_hook = None
    mutated_arg_names = inductor_meta.pop('mutated_arg_names', ())

    def decorator(fn):
        import inspect
        if 'XBLOCK' not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if 'XBLOCK' in tconfig.kwargs:
                    assert tconfig.kwargs['XBLOCK'] == 1
                    tconfig.kwargs.pop('XBLOCK')
        if config.profile_bandwidth:
            return DebugAutotuner(fn, triton_meta=triton_meta, inductor_meta=inductor_meta, regex_filter=config.profile_bandwidth_regex, configs=configs, save_cache_hook=save_cache_hook, mutated_arg_names=mutated_arg_names, heuristic_type=heuristic_type, size_hints=size_hints)
        return CachingAutotuner(fn, triton_meta=triton_meta, inductor_meta=inductor_meta, configs=configs, save_cache_hook=save_cache_hook, mutated_arg_names=mutated_arg_names, heuristic_type=heuristic_type, size_hints=size_hints)
    return decorator