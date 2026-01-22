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
def generate_stararg_copy_code(self, code):
    if not self.star_arg:
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseArgTupleInvalid', 'FunctionArguments.c'))
        code.putln('if (unlikely(%s > 0)) {' % Naming.nargs_cname)
        code.put('__Pyx_RaiseArgtupleInvalid(%s, 1, 0, 0, %s); return %s;' % (self.name.as_c_string_literal(), Naming.nargs_cname, self.error_value()))
        code.putln('}')
    if self.starstar_arg:
        if self.star_arg or not self.starstar_arg.entry.cf_used:
            kwarg_check = 'unlikely(%s)' % Naming.kwds_cname
        else:
            kwarg_check = '%s' % Naming.kwds_cname
    else:
        kwarg_check = 'unlikely(%s) && __Pyx_NumKwargs_%s(%s)' % (Naming.kwds_cname, self.signature.fastvar, Naming.kwds_cname)
    code.globalstate.use_utility_code(UtilityCode.load_cached('KeywordStringCheck', 'FunctionArguments.c'))
    code.putln('if (%s && unlikely(!__Pyx_CheckKeywordStrings(%s, %s, %d))) return %s;' % (kwarg_check, Naming.kwds_cname, self.name.as_c_string_literal(), bool(self.starstar_arg), self.error_value()))
    if self.starstar_arg and self.starstar_arg.entry.cf_used:
        code.putln('if (%s) {' % kwarg_check)
        code.putln('%s = __Pyx_KwargsAsDict_%s(%s, %s);' % (self.starstar_arg.entry.cname, self.signature.fastvar, Naming.kwds_cname, Naming.kwvalues_cname))
        code.putln('if (unlikely(!%s)) return %s;' % (self.starstar_arg.entry.cname, self.error_value()))
        code.put_gotref(self.starstar_arg.entry.cname, py_object_type)
        code.putln('} else {')
        code.putln('%s = PyDict_New();' % (self.starstar_arg.entry.cname,))
        code.putln('if (unlikely(!%s)) return %s;' % (self.starstar_arg.entry.cname, self.error_value()))
        code.put_var_gotref(self.starstar_arg.entry)
        self.starstar_arg.entry.xdecref_cleanup = False
        code.putln('}')
    if self.self_in_stararg and (not self.target.is_staticmethod):
        assert not self.signature.use_fastcall
        code.putln('%s = PyTuple_New(%s + 1); %s' % (self.star_arg.entry.cname, Naming.nargs_cname, code.error_goto_if_null(self.star_arg.entry.cname, self.pos)))
        code.put_var_gotref(self.star_arg.entry)
        code.put_incref(Naming.self_cname, py_object_type)
        code.put_giveref(Naming.self_cname, py_object_type)
        code.putln('PyTuple_SET_ITEM(%s, 0, %s);' % (self.star_arg.entry.cname, Naming.self_cname))
        temp = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
        code.putln('for (%s=0; %s < %s; %s++) {' % (temp, temp, Naming.nargs_cname, temp))
        code.putln('PyObject* item = PyTuple_GET_ITEM(%s, %s);' % (Naming.args_cname, temp))
        code.put_incref('item', py_object_type)
        code.put_giveref('item', py_object_type)
        code.putln('PyTuple_SET_ITEM(%s, %s+1, item);' % (self.star_arg.entry.cname, temp))
        code.putln('}')
        code.funcstate.release_temp(temp)
        self.star_arg.entry.xdecref_cleanup = 0
    elif self.star_arg:
        assert not self.signature.use_fastcall
        code.put_incref(Naming.args_cname, py_object_type)
        code.putln('%s = %s;' % (self.star_arg.entry.cname, Naming.args_cname))
        self.star_arg.entry.xdecref_cleanup = 0