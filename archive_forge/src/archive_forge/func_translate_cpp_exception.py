from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def translate_cpp_exception(code, pos, inside, py_result, exception_value, nogil):
    raise_py_exception, check_py_exception = get_exception_handler(exception_value)
    code.putln('try {')
    code.putln('%s' % inside)
    if py_result:
        code.putln(code.error_goto_if_null(py_result, pos))
    maybe_check_py_error(code, check_py_exception, pos, nogil)
    code.putln('} catch(...) {')
    if nogil:
        code.put_ensure_gil(declare_gilstate=True)
    code.putln(raise_py_exception)
    if nogil:
        code.put_release_ensured_gil()
    code.putln(code.error_goto(pos))
    code.putln('}')