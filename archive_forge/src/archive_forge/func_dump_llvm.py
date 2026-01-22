import atexit
import builtins
import functools
import inspect
import os
import operator
import timeit
import math
import sys
import traceback
import weakref
import warnings
import threading
import contextlib
import typing as _tp
from types import ModuleType
from importlib import import_module
import numpy as np
from inspect import signature as pysignature # noqa: F401
from inspect import Signature as pySignature # noqa: F401
from inspect import Parameter as pyParameter # noqa: F401
from numba.core.config import (PYVERSION, MACHINE_BITS, # noqa: F401
from numba.core import config
from numba.core import types
from collections.abc import Mapping, Sequence, MutableSet, MutableMapping
def dump_llvm(fndesc, module):
    print(('LLVM DUMP %s' % fndesc).center(80, '-'))
    if config.HIGHLIGHT_DUMPS:
        try:
            from pygments import highlight
            from pygments.lexers import LlvmLexer as lexer
            from pygments.formatters import Terminal256Formatter
            from numba.misc.dump_style import by_colorscheme
            print(highlight(module.__repr__(), lexer(), Terminal256Formatter(style=by_colorscheme())))
        except ImportError:
            msg = 'Please install pygments to see highlighted dumps'
            raise ValueError(msg)
    else:
        print(module)
    print('=' * 80)