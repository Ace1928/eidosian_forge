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
def generate_usr_dealloc_call(self, scope, code):
    entry = scope.lookup_here('__dealloc__')
    if not entry or not entry.is_special:
        return
    code.putln('{')
    code.putln('PyObject *etype, *eval, *etb;')
    code.putln('PyErr_Fetch(&etype, &eval, &etb);')
    code.putln('__Pyx_SET_REFCNT(o, Py_REFCNT(o) + 1);')
    code.putln('%s(o);' % entry.func_cname)
    code.putln('__Pyx_SET_REFCNT(o, Py_REFCNT(o) - 1);')
    code.putln('PyErr_Restore(etype, eval, etb);')
    code.putln('}')