import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
import importlib
import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch._subclasses import (  # noqa: F401
def is_compile_supported(device_type):
    from .eval_frame import is_dynamo_supported
    compile_supported = is_dynamo_supported()
    if device_type == 'cpu':
        pass
    elif device_type == 'cuda' and compile_supported:
        from torch.utils._triton import has_triton
        compile_supported = has_triton()
    else:
        compile_supported = False
    return compile_supported