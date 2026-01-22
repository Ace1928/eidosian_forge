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
def save_parallel_vars(self, code):
    """
        The following shenanigans are instated when we break, return or
        propagate errors from a prange. In this case we cannot rely on
        lastprivate() to do its job, as no iterations may have executed yet
        in the last thread, leaving the values undefined. It is most likely
        that the breaking thread has well-defined values of the lastprivate
        variables, so we keep those values.
        """
    section_name = '__pyx_parallel_lastprivates%d' % self.critical_section_counter
    code.putln_openmp('#pragma omp critical(%s)' % section_name)
    ParallelStatNode.critical_section_counter += 1
    code.begin_block()
    c = self.begin_of_parallel_control_block_point
    temp_count = 0
    for entry, (op, lastprivate) in sorted(self.privates.items()):
        if not lastprivate or entry.type.is_pyobject:
            continue
        if entry.type.is_cpp_class and (not entry.type.is_fake_reference) and code.globalstate.directives['cpp_locals']:
            type_decl = entry.type.cpp_optional_declaration_code('')
        else:
            type_decl = entry.type.empty_declaration_code()
        temp_cname = '__pyx_parallel_temp%d' % temp_count
        private_cname = entry.cname
        temp_count += 1
        invalid_value = entry.type.invalid_value()
        if invalid_value:
            init = ' = ' + entry.type.cast_code(invalid_value)
        else:
            init = ''
        c.putln('%s %s%s;' % (type_decl, temp_cname, init))
        self.parallel_private_temps.append((temp_cname, private_cname, entry.type))
        if entry.type.is_cpp_class:
            code.globalstate.use_utility_code(UtilityCode.load_cached('MoveIfSupported', 'CppSupport.cpp'))
            private_cname = '__PYX_STD_MOVE_IF_SUPPORTED(%s)' % private_cname
        code.putln('%s = %s;' % (temp_cname, private_cname))
    code.end_block()