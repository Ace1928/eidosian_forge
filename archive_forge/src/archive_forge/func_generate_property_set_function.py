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
def generate_property_set_function(self, property_entry, code):
    property_scope = property_entry.scope
    property_entry.setter_cname = property_scope.parent_scope.mangle(Naming.prop_set_prefix, property_entry.name)
    set_entry = property_scope.lookup_here('__set__')
    del_entry = property_scope.lookup_here('__del__')
    code.putln('')
    code.putln('static int %s(PyObject *o, PyObject *v, CYTHON_UNUSED void *x) {' % property_entry.setter_cname)
    code.putln('if (v) {')
    if set_entry:
        code.putln('return %s(o, v);' % set_entry.func_cname)
    else:
        code.putln('PyErr_SetString(PyExc_NotImplementedError, "__set__");')
        code.putln('return -1;')
    code.putln('}')
    code.putln('else {')
    if del_entry:
        code.putln('return %s(o);' % del_entry.func_cname)
    else:
        code.putln('PyErr_SetString(PyExc_NotImplementedError, "__del__");')
        code.putln('return -1;')
    code.putln('}')
    code.putln('}')