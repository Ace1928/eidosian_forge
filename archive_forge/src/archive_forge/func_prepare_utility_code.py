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
def prepare_utility_code(self):
    env = self.scope
    if env.has_import_star:
        self.create_import_star_conversion_utility_code(env)
    for name, entry in sorted(env.entries.items()):
        if entry.create_wrapper and entry.scope is env and entry.is_type and (entry.type.is_enum or entry.type.is_cpp_enum):
            entry.type.create_type_wrapper(env)