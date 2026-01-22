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
def get_annotation_info(self, signature=None):
    """
        Gets the annotation information for the function specified by
        signature. If no signature is supplied a dictionary of signature to
        annotation information is returned.
        """
    signatures = self.signatures if signature is None else [signature]
    out = collections.OrderedDict()
    for sig in signatures:
        cres = self.overloads[sig]
        ta = cres.type_annotation
        key = (ta.func_id.filename + ':' + str(ta.func_id.firstlineno + 1), ta.signature)
        out[key] = ta.annotate_raw()[key]
    return out