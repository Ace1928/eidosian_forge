import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
def error_rewrite(e, issue_type):
    """
            Rewrite and raise Exception `e` with help supplied based on the
            specified issue_type.
            """
    if config.SHOW_HELP:
        help_msg = errors.error_extras[issue_type]
        e.patch_message('\n'.join((str(e).rstrip(), help_msg)))
    if config.FULL_TRACEBACKS:
        raise e
    else:
        raise e.with_traceback(None)