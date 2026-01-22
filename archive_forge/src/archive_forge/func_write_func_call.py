from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def write_func_call(func, codewriter_class):

    def f(*args, **kwds):
        if len(args) > 1 and isinstance(args[1], codewriter_class):
            node, code = args[:2]
            marker = '                    /* %s -> %s.%s %s */' % (' ' * code.call_level, node.__class__.__name__, func.__name__, node.pos[1:])
            insertion_point = code.insertion_point()
            start = code.buffer.stream.tell()
            code.call_level += 4
            res = func(*args, **kwds)
            code.call_level -= 4
            if start != code.buffer.stream.tell():
                code.putln(marker.replace('->', '<-', 1))
                insertion_point.putln(marker)
            return res
        else:
            return func(*args, **kwds)
    return f