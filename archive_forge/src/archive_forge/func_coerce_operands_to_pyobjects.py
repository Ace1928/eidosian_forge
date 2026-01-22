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
def coerce_operands_to_pyobjects(self, env):
    self.operand2 = self.operand2.coerce_to_pyobject(env)
    if self.operand2.type is dict_type and self.operator in ('in', 'not_in'):
        self.operand2 = self.operand2.as_none_safe_node("'NoneType' object is not iterable")
    if self.cascade:
        self.cascade.coerce_operands_to_pyobjects(env)