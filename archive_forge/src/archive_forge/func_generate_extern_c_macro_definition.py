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
def generate_extern_c_macro_definition(self, code, is_cpp):
    name = Naming.extern_c_macro
    code.putln('#ifdef CYTHON_EXTERN_C')
    code.putln('    #undef %s' % name)
    code.putln('    #define %s CYTHON_EXTERN_C' % name)
    code.putln('#elif defined(%s)' % name)
    code.putln('    #ifdef _MSC_VER')
    code.putln('    #pragma message ("Please do not define the \'%s\' macro externally. Use \'CYTHON_EXTERN_C\' instead.")' % name)
    code.putln('    #else')
    code.putln("    #warning Please do not define the '%s' macro externally. Use 'CYTHON_EXTERN_C' instead." % name)
    code.putln('    #endif')
    code.putln('#else')
    if is_cpp:
        code.putln('    #define %s extern "C++"' % name)
    else:
        code.putln('  #ifdef __cplusplus')
        code.putln('    #define %s extern "C"' % name)
        code.putln('  #else')
        code.putln('    #define %s extern' % name)
        code.putln('  #endif')
    code.putln('#endif')