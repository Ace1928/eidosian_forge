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
def process_implementation(self, options, result):
    env = self.scope
    env.return_type = PyrexTypes.c_void_type
    self.referenced_modules = []
    self.find_referenced_modules(env, self.referenced_modules, {})
    self.sort_cdef_classes(env)
    self.generate_c_code(env, options, result)
    self.generate_h_code(env, options, result)
    self.generate_api_code(env, options, result)