from __future__ import annotations
import base64
import copyreg
import dataclasses
import functools
import hashlib
import importlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import pickle
import pkgutil
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import warnings
import weakref
from bisect import bisect_right
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from ctypes import c_void_p, cdll, CDLL
from dataclasses import field
from functools import partial
from importlib import abc
from pathlib import Path
from threading import Thread
from time import sleep, time
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
import torch
from torch._dynamo.device_interface import (
from torch._dynamo.utils import counters
from torch._inductor import config, exc
from torch._inductor.codegen.cuda import cuda_env
from torch._inductor.utils import cache_dir, developer_warning, is_linux
from torch._prims_common import suggest_memory_format
from torch.fx.experimental.symbolic_shapes import has_hint, hint_int, ShapeEnv
from torch.hub import _Faketqdm, tqdm
import torch
from ctypes import cdll
def cuda_compile_command(src_files: List[str], dst_file: str, dst_file_ext: str) -> str:
    include_paths = _cutlass_include_paths()
    cuda_lib_options = _cuda_lib_options()
    nvcc_host_compiler_options = _nvcc_host_compiler_options()
    nvcc_compiler_options = _nvcc_compiler_options()
    options = nvcc_compiler_options + [f'-Xcompiler {opt}' if '=' in opt else f'-Xcompiler={opt}' for opt in nvcc_host_compiler_options] + ['-I' + path for path in include_paths] + cuda_lib_options
    src_file = ' '.join(src_files)
    res = ''
    if dst_file_ext == 'o':
        res = f'{_cuda_compiler()} {' '.join(options)} -c -o {dst_file} {src_file}'
    elif dst_file_ext == 'so':
        options.append('-shared')
        res = f'{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}'
    else:
        raise NotImplementedError(f'Unsupported output file suffix {dst_file_ext}!')
    log.debug('CUDA command: %s', res)
    return res