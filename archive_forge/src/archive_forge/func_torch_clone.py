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
def torch_clone(x):
    y = torch.clone(x)
    if x.is_leaf:
        y.requires_grad_(x.requires_grad)
    if x.is_leaf and x.grad is not None:
        y.grad = clone_input(x.grad, dtype=dtype)
    if hasattr(x, '_dynamo_dynamic_indices'):
        y._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()
    return y