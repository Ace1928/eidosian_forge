import ast
import copy
import functools
import linecache
import sys
from typing import Any, Dict, List
import triton

    Specializes a triton kernel with variable number of inputs
    to a specific number of inputs `N`.
    NOTE: Because it's quite costly to call `triton.jit`,
    we cache the returned value with `lru_cache`
    