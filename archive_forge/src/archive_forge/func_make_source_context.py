import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, List, NamedTuple, Optional, Tuple
from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory
@functools.lru_cache(maxsize=None)
def make_source_context(*args):
    return SourceContext(*args)