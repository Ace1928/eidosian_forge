from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
def put_assign_to_buffer(lhs_cname, rhs_cname, buf_entry, is_initialized, pos, code):
    """
    Generate code for reassigning a buffer variables. This only deals with getting
    the buffer auxiliary structure and variables set up correctly, the assignment
    itself and refcounting is the responsibility of the caller.

    However, the assignment operation may throw an exception so that the reassignment
    never happens.

    Depending on the circumstances there are two possible outcomes:
    - Old buffer released, new acquired, rhs assigned to lhs
    - Old buffer released, new acquired which fails, reaqcuire old lhs buffer
      (which may or may not succeed).
    """
    buffer_aux, buffer_type = (buf_entry.buffer_aux, buf_entry.type)
    pybuffernd_struct = buffer_aux.buflocal_nd_var.cname
    flags = get_flags(buffer_aux, buffer_type)
    code.putln('{')
    code.putln('__Pyx_BufFmt_StackElem __pyx_stack[%d];' % buffer_type.dtype.struct_nesting_depth())
    getbuffer = get_getbuffer_call(code, '%s', buffer_aux, buffer_type)
    if is_initialized:
        code.putln('__Pyx_SafeReleaseBuffer(&%s.rcbuffer->pybuffer);' % pybuffernd_struct)
        retcode_cname = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
        code.putln('%s = %s;' % (retcode_cname, getbuffer % rhs_cname))
        code.putln('if (%s) {' % code.unlikely('%s < 0' % retcode_cname))
        exc_temps = tuple((code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=False) for _ in range(3)))
        code.putln('PyErr_Fetch(&%s, &%s, &%s);' % exc_temps)
        code.putln('if (%s) {' % code.unlikely('%s == -1' % (getbuffer % lhs_cname)))
        code.putln('Py_XDECREF(%s); Py_XDECREF(%s); Py_XDECREF(%s);' % exc_temps)
        code.globalstate.use_utility_code(raise_buffer_fallback_code)
        code.putln('__Pyx_RaiseBufferFallbackError();')
        code.putln('} else {')
        code.putln('PyErr_Restore(%s, %s, %s);' % exc_temps)
        code.putln('}')
        code.putln('%s = %s = %s = 0;' % exc_temps)
        for t in exc_temps:
            code.funcstate.release_temp(t)
        code.putln('}')
        put_unpack_buffer_aux_into_scope(buf_entry, code)
        code.putln(code.error_goto_if_neg(retcode_cname, pos))
        code.funcstate.release_temp(retcode_cname)
    else:
        code.putln('if (%s) {' % code.unlikely('%s == -1' % (getbuffer % rhs_cname)))
        code.putln('%s = %s; __Pyx_INCREF(Py_None); %s.rcbuffer->pybuffer.buf = NULL;' % (lhs_cname, PyrexTypes.typecast(buffer_type, PyrexTypes.py_object_type, 'Py_None'), pybuffernd_struct))
        code.putln(code.error_goto(pos))
        code.put('} else {')
        put_unpack_buffer_aux_into_scope(buf_entry, code)
        code.putln('}')
    code.putln('}')