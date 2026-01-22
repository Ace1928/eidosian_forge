import ast
from collections import defaultdict
import errno
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import itertools
import marshal
import os
from pathlib import Path
from pathlib import PurePath
import struct
import sys
import tokenize
import types
from typing import Callable
from typing import Dict
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from _pytest._io.saferepr import DEFAULT_REPR_MAX_SIZE
from _pytest._io.saferepr import saferepr
from _pytest._version import version
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.main import Session
from _pytest.pathlib import absolutepath
from _pytest.pathlib import fnmatch_ex
from _pytest.stash import StashKey
from _pytest.assertion.util import format_explanation as _format_explanation  # noqa:F401, isort:skip
def pop_format_context(self, expl_expr: ast.expr) -> ast.Name:
    """Format the %-formatted string with current format context.

        The expl_expr should be an str ast.expr instance constructed from
        the %-placeholders created by .explanation_param().  This will
        add the required code to format said string to .expl_stmts and
        return the ast.Name instance of the formatted string.
        """
    current = self.stack.pop()
    if self.stack:
        self.explanation_specifiers = self.stack[-1]
    keys = [ast.Constant(key) for key in current.keys()]
    format_dict = ast.Dict(keys, list(current.values()))
    form = ast.BinOp(expl_expr, ast.Mod(), format_dict)
    name = '@py_format' + str(next(self.variable_counter))
    if self.enable_assertion_pass_hook:
        self.format_variables.append(name)
    self.expl_stmts.append(ast.Assign([ast.Name(name, ast.Store())], form))
    return ast.Name(name, ast.Load())