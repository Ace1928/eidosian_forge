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
def sort_types_by_inheritance(self, type_dict, type_order, getkey):
    subclasses = defaultdict(list)
    for key in type_order:
        new_entry = type_dict[key]
        base = new_entry.type.base_type
        while base:
            base_key = getkey(base)
            subclasses[base_key].append(key)
            base_entry = type_dict.get(base_key)
            if base_entry is None:
                break
            base = base_entry.type.base_type
    seen = set()
    result = []

    def dfs(u):
        if u in seen:
            return
        seen.add(u)
        for v in subclasses[getkey(u.type)]:
            dfs(type_dict[v])
        result.append(u)
    for key in reversed(type_order):
        dfs(type_dict[key])
    result.reverse()
    return result