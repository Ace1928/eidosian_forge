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
def generate_cclass_include_code(self, type, i_code):
    i_code.putln('cdef extern class %s.%s:' % (type.module_name, type.name))
    i_code.indent()
    var_entries = type.scope.var_entries
    if var_entries:
        for entry in var_entries:
            i_code.putln('cdef %s' % entry.type.declaration_code(entry.cname, pyrex=1))
    else:
        i_code.putln('pass')
    i_code.dedent()