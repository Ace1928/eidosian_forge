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
def generate_descr_get_function(self, scope, code):
    user_get_entry = scope.lookup_here('__get__')
    code.putln('')
    code.putln('static PyObject *%s(PyObject *o, PyObject *i, PyObject *c) {' % scope.mangle_internal('tp_descr_get'))
    code.putln('PyObject *r = 0;')
    code.putln('if (!i) i = Py_None;')
    code.putln('if (!c) c = Py_None;')
    code.putln('r = %s(o, i, c);' % user_get_entry.func_cname)
    code.putln('return r;')
    code.putln('}')