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
def lookup_here_or_base(n, tp=None, extern_return=None):
    if tp is None:
        tp = scope.parent_type
    r = tp.scope.lookup_here(n)
    if r is None:
        if tp.is_external and extern_return is not None:
            return extern_return
        if tp.base_type is not None:
            return lookup_here_or_base(n, tp.base_type)
    return r