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
def generate_argument_parsing_code(self, env, code, decl_code):
    old_error_label = code.new_error_label()
    our_error_label = code.error_label
    end_label = code.new_label('argument_unpacking_done')
    has_kwonly_args = self.num_kwonly_args > 0
    has_star_or_kw_args = self.star_arg is not None or self.starstar_arg is not None or has_kwonly_args
    for arg in self.args:
        if not arg.type.is_pyobject:
            if not arg.type.create_from_py_utility_code(env):
                pass
    if self.signature_has_generic_args():
        if self.signature.use_fastcall:
            code.putln('#if !CYTHON_METH_FASTCALL')
        code.putln('#if CYTHON_ASSUME_SAFE_MACROS')
        code.putln('%s = PyTuple_GET_SIZE(%s);' % (Naming.nargs_cname, Naming.args_cname))
        code.putln('#else')
        code.putln('%s = PyTuple_Size(%s); if (%s) return %s;' % (Naming.nargs_cname, Naming.args_cname, code.unlikely('%s < 0' % Naming.nargs_cname), self.error_value()))
        code.putln('#endif')
        if self.signature.use_fastcall:
            code.putln('#endif')
    code.globalstate.use_utility_code(UtilityCode.load_cached('fastcall', 'FunctionArguments.c'))
    code.putln('%s = __Pyx_KwValues_%s(%s, %s);' % (Naming.kwvalues_cname, self.signature.fastvar, Naming.args_cname, Naming.nargs_cname))
    if not self.signature_has_generic_args():
        if has_star_or_kw_args:
            error(self.pos, 'This method cannot have * or keyword arguments')
        self.generate_argument_conversion_code(code)
    elif not self.signature_has_nongeneric_args():
        self.generate_stararg_copy_code(code)
    else:
        self.generate_tuple_and_keyword_parsing_code(self.args, code, decl_code)
        self.needs_values_cleanup = True
    code.error_label = old_error_label
    if code.label_used(our_error_label):
        if not code.label_used(end_label):
            code.put_goto(end_label)
        code.put_label(our_error_label)
        self.generate_argument_values_cleanup_code(code)
        if has_star_or_kw_args:
            self.generate_arg_decref(self.star_arg, code)
            if self.starstar_arg:
                if self.starstar_arg.entry.xdecref_cleanup:
                    code.put_var_xdecref_clear(self.starstar_arg.entry)
                else:
                    code.put_var_decref_clear(self.starstar_arg.entry)
        for arg in self.args:
            if not arg.type.is_pyobject and arg.type.needs_refcounting:
                code.put_var_xdecref(arg.entry)
        code.put_add_traceback(self.target.entry.qualified_name)
        code.put_finish_refcount_context()
        code.putln('return %s;' % self.error_value())
    if code.label_used(end_label):
        code.put_label(end_label)