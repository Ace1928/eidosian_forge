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
def assert_no_fake_params_or_buffers(gm):
    from torch._subclasses.fake_tensor import FakeTensorConfig

    def stack_or_hint(t):
        if FakeTensorConfig.debug:
            import traceback
            return f'FAKE TENSOR CREATION TRACEBACK: \n {traceback.format_list(t._debug_trace)}'
        else:
            return 'Enable TORCH_FAKE_TENSOR_DEBUG=1 to get creation stack traces on fake tensors.'
    for name, buffer in gm.named_buffers():
        assert not isinstance(buffer, torch._subclasses.FakeTensor), f'Unexpected fake buffer {name} {stack_or_hint(buffer)}'
    for name, param in gm.named_parameters():
        assert not isinstance(param, torch._subclasses.FakeTensor), f'Unexpected fake param {name} {stack_or_hint(param)}'