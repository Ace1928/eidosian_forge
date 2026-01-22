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
def get_include_and_linking_paths(include_pytorch: bool=False, vec_isa: VecISA=invalid_vec_isa, cuda: bool=False, aot_mode: bool=False) -> Tuple[List[str], str, str, str, str]:
    if config.is_fbcode() and 'CUDA_HOME' not in os.environ and ('CUDA_PATH' not in os.environ):
        os.environ['CUDA_HOME'] = os.path.dirname(build_paths.cuda())
    from torch.utils import cpp_extension
    macros = ''
    build_arch_flags = ''
    if sys.platform == 'linux' and (include_pytorch or vec_isa != invalid_vec_isa or cuda or config.cpp.enable_kernel_profile):
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path('include')]
        lpaths = cpp_extension.library_paths(cuda) + [sysconfig.get_config_var('LIBDIR')]
        libs = []
        if not config.is_fbcode():
            libs += ['torch', 'torch_cpu']
            libs += ['gomp']
            if not aot_mode:
                libs += ['torch_python']
        else:
            libs += ['omp']
            if aot_mode:
                ipaths += [os.path.dirname(cpp_prefix_path())]
                if cuda:
                    for i, path in enumerate(lpaths):
                        if path.startswith(os.environ['CUDA_HOME']) and (not os.path.exists(f'{path}/libcudart_static.a')):
                            for root, dirs, files in os.walk(path):
                                if 'libcudart_static.a' in files:
                                    lpaths[i] = os.path.join(path, root)
                                    lpaths.append(os.path.join(lpaths[i], 'stubs'))
                                    break
        macros = vec_isa.build_macro()
        if macros:
            if config.is_fbcode() and vec_isa != invalid_vec_isa:
                cap = str(vec_isa).upper()
                macros = ' '.join([vec_isa.build_arch_flags(), f'-D CPU_CAPABILITY={cap}', f'-D CPU_CAPABILITY_{cap}', f'-D HAVE_{cap}_CPU_DEFINITION'])
        if aot_mode and cuda:
            if macros is None:
                macros = ''
            macros += ' -D USE_CUDA'
        if cuda:
            if torch.version.hip is not None:
                libs += ['c10_hip', 'torch_hip']
            elif config.is_fbcode():
                libs += ['cuda']
            else:
                libs += ['c10_cuda', 'cuda', 'torch_cuda']
        build_arch_flags = vec_isa.build_arch_flags()
    else:
        ipaths = cpp_extension.include_paths(cuda) + [sysconfig.get_path('include')]
        if aot_mode:
            ipaths += [os.path.dirname(cpp_prefix_path())]
        lpaths = []
        if sys.platform == 'darwin':
            omp_available = not is_apple_clang()
            if os.getenv('OMP_PREFIX') is not None:
                header_path = os.path.join(os.getenv('OMP_PREFIX'), 'include', 'omp.h')
                valid_env = os.path.exists(header_path)
                if valid_env:
                    ipaths.append(os.path.join(os.getenv('OMP_PREFIX'), 'include'))
                    lpaths.append(os.path.join(os.getenv('OMP_PREFIX'), 'lib'))
                else:
                    warnings.warn('environment variable `OMP_PREFIX` is invalid.')
                omp_available = omp_available or valid_env
            libs = [] if omp_available else ['omp']
            if not omp_available and os.getenv('CONDA_PREFIX') is not None:
                omp_available = is_conda_llvm_openmp_installed()
                if omp_available:
                    conda_lib_path = os.path.join(os.getenv('CONDA_PREFIX'), 'lib')
                    ipaths.append(os.path.join(os.getenv('CONDA_PREFIX'), 'include'))
                    lpaths.append(conda_lib_path)
                    if os.uname().machine == 'x86_64' and os.path.exists(os.path.join(conda_lib_path, 'libiomp5.dylib')):
                        libs = ['iomp5']
            if not omp_available:
                omp_available, libomp_path = homebrew_libomp()
                if omp_available:
                    ipaths.append(os.path.join(libomp_path, 'include'))
                    lpaths.append(os.path.join(libomp_path, 'lib'))
        else:
            libs = ['omp'] if config.is_fbcode() else ['gomp']
    if not config.aot_inductor.abi_compatible:
        libs += ['c10']
        lpaths += [cpp_extension.TORCH_LIB_PATH]
    if config.is_fbcode():
        ipaths.append(build_paths.sleef())
        ipaths.append(build_paths.openmp())
        ipaths.append(build_paths.cc_include())
        ipaths.append(build_paths.libgcc())
        ipaths.append(build_paths.libgcc_arch())
        ipaths.append(build_paths.libgcc_backward())
        ipaths.append(build_paths.glibc())
        ipaths.append(build_paths.linux_kernel())
        ipaths.append(build_paths.cuda())
        ipaths.append('include')
    static_link_libs = []
    if aot_mode and cuda and config.is_fbcode():
        static_link_libs = ['-Wl,-Bstatic', '-lcudart_static', '-Wl,-Bdynamic']
    lpaths_str = ' '.join(['-L' + p for p in lpaths])
    libs_str = ' '.join(static_link_libs + ['-l' + p for p in libs])
    return (ipaths, lpaths_str, libs_str, macros, build_arch_flags)