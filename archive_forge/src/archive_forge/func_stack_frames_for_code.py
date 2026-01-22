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
@classmethod
@functools.lru_cache(None)
def stack_frames_for_code(cls, path: str, lineno: int) -> Optional[List[Dict[str, Any]]]:
    if path not in cls.linemaps:
        return None
    lines, nodes = cls.linemaps[path]
    p = bisect_right(lines, lineno)
    if p == 0:
        return None
    entry = nodes[p - 1]
    if not entry:
        return None

    def parse_stack_trace(stack_trace: str) -> List[Dict[str, Any]]:
        regex = 'File "(.+)", line (\\d+), in (.+)\\n'
        matches = re.findall(regex, stack_trace)
        return [{'filename': f, 'line': int(l), 'name': n} for f, l, n in reversed(matches)]
    return parse_stack_trace(entry)