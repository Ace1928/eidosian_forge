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
def generate_method_table(self, env, code):
    if env.is_c_class_scope and (not env.pyfunc_entries):
        return
    binding = env.directives['binding']
    code.putln('')
    wrapper_code_writer = code.insertion_point()
    code.putln('static PyMethodDef %s[] = {' % env.method_table_cname)
    for entry in env.pyfunc_entries:
        if not entry.fused_cfunction and (not (binding and entry.is_overridable)):
            code.put_pymethoddef(entry, ',', wrapper_code_writer=wrapper_code_writer)
    code.putln('{0, 0, 0, 0}')
    code.putln('};')
    if wrapper_code_writer.getvalue():
        wrapper_code_writer.putln('')