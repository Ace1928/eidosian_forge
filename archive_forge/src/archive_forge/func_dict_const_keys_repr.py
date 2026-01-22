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
def dict_const_keys_repr(const_keys, *, local):
    if any((isinstance(k, enum.Enum) for k in const_keys)):
        const_keys_str = f'{ {enum_repr(k, local=local) if isinstance(k, enum.Enum) else repr(k) for k in const_keys}}'.replace("'", '')
    else:
        const_keys_str = f'{const_keys!r}'
    return const_keys_str