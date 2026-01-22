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
def generate_generic_parallel_unpacking_code(self, code, rhs, unpacked_items, use_loop, terminate=True):
    code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
    code.globalstate.use_utility_code(UtilityCode.load_cached('IterFinish', 'ObjectHandling.c'))
    code.putln('Py_ssize_t index = -1;')
    if use_loop:
        code.putln('PyObject** temps[%s] = {%s};' % (len(self.unpacked_items), ','.join(['&%s' % item.result() for item in unpacked_items])))
    iterator_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
    code.putln('%s = PyObject_GetIter(%s); %s' % (iterator_temp, rhs.py_result(), code.error_goto_if_null(iterator_temp, self.pos)))
    code.put_gotref(iterator_temp, py_object_type)
    rhs.generate_disposal_code(code)
    iternext_func = code.funcstate.allocate_temp(self._func_iternext_type, manage_ref=False)
    code.putln('%s = __Pyx_PyObject_GetIterNextFunc(%s);' % (iternext_func, iterator_temp))
    unpacking_error_label = code.new_label('unpacking_failed')
    unpack_code = '%s(%s)' % (iternext_func, iterator_temp)
    if use_loop:
        code.putln('for (index=0; index < %s; index++) {' % len(unpacked_items))
        code.put('PyObject* item = %s; if (unlikely(!item)) ' % unpack_code)
        code.put_goto(unpacking_error_label)
        code.put_gotref('item', py_object_type)
        code.putln('*(temps[index]) = item;')
        code.putln('}')
    else:
        for i, item in enumerate(unpacked_items):
            code.put('index = %d; %s = %s; if (unlikely(!%s)) ' % (i, item.result(), unpack_code, item.result()))
            code.put_goto(unpacking_error_label)
            item.generate_gotref(code)
    if terminate:
        code.globalstate.use_utility_code(UtilityCode.load_cached('UnpackItemEndCheck', 'ObjectHandling.c'))
        code.put_error_if_neg(self.pos, '__Pyx_IternextUnpackEndCheck(%s, %d)' % (unpack_code, len(unpacked_items)))
        code.putln('%s = NULL;' % iternext_func)
        code.put_decref_clear(iterator_temp, py_object_type)
    unpacking_done_label = code.new_label('unpacking_done')
    code.put_goto(unpacking_done_label)
    code.put_label(unpacking_error_label)
    code.put_decref_clear(iterator_temp, py_object_type)
    code.putln('%s = NULL;' % iternext_func)
    code.putln('if (__Pyx_IterFinish() == 0) __Pyx_RaiseNeedMoreValuesError(index);')
    code.putln(code.error_goto(self.pos))
    code.put_label(unpacking_done_label)
    code.funcstate.release_temp(iternext_func)
    if terminate:
        code.funcstate.release_temp(iterator_temp)
        iterator_temp = None
    return iterator_temp