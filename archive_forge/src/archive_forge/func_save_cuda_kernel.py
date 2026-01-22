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
def save_cuda_kernel(self, grid, stream, launcher):
    if callable(grid):
        grid_x, grid_y, grid_z = grid(launcher.config.kwargs)
    else:
        grid_x, grid_y, grid_z = grid
    key = self.inductor_meta.get('kernel_name', None)
    assert key is not None, 'kernel_name can not be None'
    params = {'mangled_name': launcher.bin.metadata['name'], 'grid_x': grid_x, 'grid_y': grid_y, 'grid_z': grid_z, 'x_block': launcher.config.kwargs.get('XBLOCK', 1), 'y_block': launcher.config.kwargs.get('YBLOCK', None), 'z_block': launcher.config.kwargs.get('ZBLOCK', None), 'num_warps': launcher.bin.num_warps, 'shared_mem': launcher.bin.shared, 'stream': stream, 'meta': launcher.config.kwargs}
    if torch.version.hip is None:
        CudaKernelParamCache.set(key, params, launcher.bin.asm['cubin'])
    else:
        import pathlib
        launcher.bin.asm['hsaco'] = pathlib.Path(launcher.bin.asm['hsaco_path']).read_bytes()
        CudaKernelParamCache.set(key, params, launcher.bin.asm['hsaco'])