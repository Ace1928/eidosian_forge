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
def generate_operation_code(self, code, result_code, operand1, op, operand2):
    if self.type.is_pyobject:
        error_clause = code.error_goto_if_null
        got_ref = '__Pyx_XGOTREF(%s); ' % result_code
        if self.special_bool_cmp_function:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyBoolOrNullFromLong', 'ObjectHandling.c'))
            coerce_result = '__Pyx_PyBoolOrNull_FromLong'
        else:
            coerce_result = '__Pyx_PyBool_FromLong'
    else:
        error_clause = code.error_goto_if_neg
        got_ref = ''
        coerce_result = ''
    if self.special_bool_cmp_function:
        if operand1.type.is_pyobject:
            result1 = operand1.py_result()
        else:
            result1 = operand1.result()
        if operand2.type.is_pyobject:
            result2 = operand2.py_result()
        else:
            result2 = operand2.result()
        special_bool_extra_args_result = ', '.join([extra_arg.result() for extra_arg in self.special_bool_extra_args])
        if self.special_bool_cmp_utility_code:
            code.globalstate.use_utility_code(self.special_bool_cmp_utility_code)
        code.putln('%s = %s(%s(%s, %s, %s)); %s%s' % (result_code, coerce_result, self.special_bool_cmp_function, result1, result2, special_bool_extra_args_result if self.special_bool_extra_args else richcmp_constants[op], got_ref, error_clause(result_code, self.pos)))
    elif operand1.type.is_pyobject and op not in ('is', 'is_not'):
        assert op not in ('in', 'not_in'), op
        assert self.type.is_pyobject or self.type is PyrexTypes.c_bint_type
        code.putln('%s = PyObject_RichCompare%s(%s, %s, %s); %s%s' % (result_code, '' if self.type.is_pyobject else 'Bool', operand1.py_result(), operand2.py_result(), richcmp_constants[op], got_ref, error_clause(result_code, self.pos)))
    elif operand1.type.is_complex:
        code.putln('%s = %s(%s%s(%s, %s));' % (result_code, coerce_result, op == '!=' and '!' or '', operand1.type.unary_op('eq'), operand1.result(), operand2.result()))
    else:
        type1 = operand1.type
        type2 = operand2.type
        if (type1.is_extension_type or type2.is_extension_type) and (not type1.same_as(type2)):
            common_type = py_object_type
        elif type1.is_numeric:
            common_type = PyrexTypes.widest_numeric_type(type1, type2)
        else:
            common_type = type1
        code1 = operand1.result_as(common_type)
        code2 = operand2.result_as(common_type)
        statement = '%s = %s(%s %s %s);' % (result_code, coerce_result, code1, self.c_operator(op), code2)
        if self.is_cpp_comparison() and self.exception_check == '+':
            translate_cpp_exception(code, self.pos, statement, result_code if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
        else:
            code.putln(statement)