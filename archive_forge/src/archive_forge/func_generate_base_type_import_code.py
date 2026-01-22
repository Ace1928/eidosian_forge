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
def generate_base_type_import_code(self, env, entry, code, import_generator):
    base_type = entry.type.base_type
    if base_type and base_type.module_name != env.qualified_name and (not (base_type.is_builtin_type or base_type.is_cython_builtin_type)) and (not entry.utility_code_definition):
        self.generate_type_import_code(env, base_type, self.pos, code, import_generator)