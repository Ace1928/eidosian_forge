import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, List, NamedTuple, Optional, Tuple
from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory
def fake_range():
    return SourceContext('', None, 0, 0).make_raw_range(0, 1)