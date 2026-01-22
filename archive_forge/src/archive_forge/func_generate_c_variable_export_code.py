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
def generate_c_variable_export_code(self, env, code):
    entries = []
    for entry in env.var_entries:
        if entry.api or entry.defined_in_pxd or (Options.cimport_from_pyx and (not entry.visibility == 'extern')):
            entries.append(entry)
    if entries:
        env.use_utility_code(UtilityCode.load_cached('VoidPtrExport', 'ImportExport.c'))
        for entry in entries:
            signature = entry.type.empty_declaration_code()
            name = code.intern_identifier(entry.name)
            code.putln('if (__Pyx_ExportVoidPtr(%s, (void *)&%s, "%s") < 0) %s' % (name, entry.cname, signature, code.error_goto(self.pos)))