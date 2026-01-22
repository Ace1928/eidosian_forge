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
def generate_c_function_import_code_for_module(self, module, env, code):
    entries = []
    for entry in module.cfunc_entries:
        if entry.defined_in_pxd and entry.used:
            entries.append(entry)
    if entries:
        env.use_utility_code(UtilityCode.load_cached('FunctionImport', 'ImportExport.c'))
        temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        code.putln('%s = PyImport_ImportModule("%s"); if (!%s) %s' % (temp, module.qualified_name, temp, code.error_goto(self.pos)))
        code.put_gotref(temp, py_object_type)
        for entry in entries:
            code.putln('if (__Pyx_ImportFunction_%s(%s, %s, (void (**)(void))&%s, "%s") < 0) %s' % (Naming.cyversion, temp, entry.name.as_c_string_literal(), entry.cname, entry.type.signature_string(), code.error_goto(self.pos)))
        code.put_decref_clear(temp, py_object_type)
        code.funcstate.release_temp(temp)