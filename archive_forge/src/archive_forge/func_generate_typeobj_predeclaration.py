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
def generate_typeobj_predeclaration(self, entry, code):
    code.putln('')
    name = entry.type.typeobj_cname
    if name:
        if entry.visibility == 'extern' and (not entry.in_cinclude):
            code.putln('%s %s %s;' % (Naming.extern_c_macro, PyrexTypes.public_decl('PyTypeObject', 'DL_IMPORT'), name))
        elif entry.visibility == 'public':
            code.putln('%s %s %s;' % (Naming.extern_c_macro, PyrexTypes.public_decl('PyTypeObject', 'DL_EXPORT'), name))