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
def coerce_to(self, dst_type, env):
    if dst_type == self.assignment.rhs.type:
        old_rhs_arg = self.rhs.arg
        if isinstance(old_rhs_arg, CoerceToTempNode):
            old_rhs_arg = old_rhs_arg.arg
        rhs_arg = old_rhs_arg.coerce_to(dst_type, env)
        if rhs_arg is not old_rhs_arg:
            self.rhs.arg = rhs_arg
            self.rhs.update_type_and_entry()
            if isinstance(self.assignment.rhs, CoercionNode) and (not isinstance(self.assignment.rhs, CloneNode)):
                self.assignment.rhs = self.assignment.rhs.arg
                self.assignment.rhs.type = self.assignment.rhs.arg.type
            return self
    return super(AssignmentExpressionNode, self).coerce_to(dst_type, env)