import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
def heuristic_record_if_ctx_manager(obj, module, name):
    if issubclass(type(obj), type) and hasattr(obj, '__enter__') and hasattr(obj, '__exit__'):
        torch_name_rule_map[f'{module.__name__}.{name}'] = TorchCtxManagerClassVariable
        ctx_mamager_classes.add(obj)