from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def generate_getattro_function(self, scope, code):

    def lookup_here_or_base(n, tp=None, extern_return=None):
        if tp is None:
            tp = scope.parent_type
        r = tp.scope.lookup_here(n)
        if r is None:
            if tp.is_external and extern_return is not None:
                return extern_return
            if tp.base_type is not None:
                return lookup_here_or_base(n, tp.base_type)
        return r
    has_instance_dict = lookup_here_or_base('__dict__', extern_return='extern')
    getattr_entry = lookup_here_or_base('__getattr__')
    getattribute_entry = lookup_here_or_base('__getattribute__')
    code.putln('')
    code.putln('static PyObject *%s(PyObject *o, PyObject *n) {' % scope.mangle_internal('tp_getattro'))
    if getattribute_entry is not None:
        code.putln('PyObject *v = %s(o, n);' % getattribute_entry.func_cname)
    else:
        if not has_instance_dict and scope.parent_type.is_final_type:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObject_GenericGetAttrNoDict', 'ObjectHandling.c'))
            generic_getattr_cfunc = '__Pyx_PyObject_GenericGetAttrNoDict'
        elif not has_instance_dict or has_instance_dict == 'extern':
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObject_GenericGetAttr', 'ObjectHandling.c'))
            generic_getattr_cfunc = '__Pyx_PyObject_GenericGetAttr'
        else:
            generic_getattr_cfunc = 'PyObject_GenericGetAttr'
        code.putln('PyObject *v = %s(o, n);' % generic_getattr_cfunc)
    if getattr_entry is not None:
        code.putln('if (!v && PyErr_ExceptionMatches(PyExc_AttributeError)) {')
        code.putln('PyErr_Clear();')
        code.putln('v = %s(o, n);' % getattr_entry.func_cname)
        code.putln('}')
    code.putln('return v;')
    code.putln('}')