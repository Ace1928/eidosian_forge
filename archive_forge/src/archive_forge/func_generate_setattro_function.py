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
def generate_setattro_function(self, scope, code):
    base_type = scope.parent_type.base_type
    set_entry = scope.lookup_here('__setattr__')
    del_entry = scope.lookup_here('__delattr__')
    code.putln('')
    code.putln('static int %s(PyObject *o, PyObject *n, PyObject *v) {' % scope.mangle_internal('tp_setattro'))
    code.putln('if (v) {')
    if set_entry:
        code.putln('return %s(o, n, v);' % set_entry.func_cname)
    else:
        self.generate_guarded_basetype_call(base_type, None, 'tp_setattro', 'o, n, v', code)
        code.putln('return PyObject_GenericSetAttr(o, n, v);')
    code.putln('}')
    code.putln('else {')
    if del_entry:
        code.putln('return %s(o, n);' % del_entry.func_cname)
    else:
        self.generate_guarded_basetype_call(base_type, None, 'tp_setattro', 'o, n, v', code)
        code.putln('return PyObject_GenericSetAttr(o, n, 0);')
    code.putln('}')
    code.putln('}')