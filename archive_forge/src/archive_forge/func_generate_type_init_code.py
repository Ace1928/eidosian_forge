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
def generate_type_init_code(self, env, code):
    with ModuleImportGenerator(code) as import_generator:
        for entry in env.c_class_entries:
            if entry.visibility == 'extern' and (not entry.utility_code_definition):
                self.generate_type_import_code(env, entry.type, entry.pos, code, import_generator)
            else:
                self.generate_base_type_import_code(env, entry, code, import_generator)
                self.generate_exttype_vtable_init_code(entry, code)
                if entry.type.early_init:
                    self.generate_type_ready_code(entry, code)