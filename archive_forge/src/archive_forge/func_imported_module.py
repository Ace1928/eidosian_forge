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
def imported_module(self, module_name_string, error_code):
    if module_name_string in self.imported:
        return self.imported[module_name_string]
    code = self.code
    temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
    self.temps.append(temp)
    code.putln('%s = PyImport_ImportModule(%s); if (unlikely(!%s)) %s' % (temp, module_name_string, temp, error_code))
    code.put_gotref(temp, py_object_type)
    self.imported[module_name_string] = temp
    return temp