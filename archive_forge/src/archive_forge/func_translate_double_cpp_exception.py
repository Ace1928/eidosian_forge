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
def translate_double_cpp_exception(code, pos, lhs_type, lhs_code, rhs_code, lhs_exc_val, assign_exc_val, nogil):
    handle_lhs_exc, lhc_check_py_exc = get_exception_handler(lhs_exc_val)
    handle_assignment_exc, assignment_check_py_exc = get_exception_handler(assign_exc_val)
    code.putln('try {')
    code.putln(lhs_type.declaration_code('__pyx_local_lvalue = %s;' % lhs_code))
    maybe_check_py_error(code, lhc_check_py_exc, pos, nogil)
    code.putln('try {')
    code.putln('__pyx_local_lvalue = %s;' % rhs_code)
    maybe_check_py_error(code, assignment_check_py_exc, pos, nogil)
    code.putln('} catch(...) {')
    if nogil:
        code.put_ensure_gil(declare_gilstate=True)
    code.putln(handle_assignment_exc)
    if nogil:
        code.put_release_ensured_gil()
    code.putln(code.error_goto(pos))
    code.putln('}')
    code.putln('} catch(...) {')
    if nogil:
        code.put_ensure_gil(declare_gilstate=True)
    code.putln(handle_lhs_exc)
    if nogil:
        code.put_release_ensured_gil()
    code.putln(code.error_goto(pos))
    code.putln('}')