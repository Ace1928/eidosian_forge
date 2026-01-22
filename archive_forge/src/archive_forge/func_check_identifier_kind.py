from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def check_identifier_kind(self):
    entry = self.entry
    if entry.is_type and entry.type.is_extension_type:
        self.type_entry = entry
    if entry.is_type and (entry.type.is_enum or entry.type.is_cpp_enum):
        py_entry = Symtab.Entry(self.name, None, py_object_type)
        py_entry.is_pyglobal = True
        py_entry.scope = self.entry.scope
        self.entry = py_entry
    elif not (entry.is_const or entry.is_variable or entry.is_builtin or entry.is_cfunction or entry.is_cpp_class):
        if self.entry.as_variable:
            self.entry = self.entry.as_variable
        elif not self.is_cython_module:
            error(self.pos, "'%s' is not a constant, variable or function identifier" % self.name)