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
def calculate_is_sequence_mul(self):
    type1 = self.operand1.type
    type2 = self.operand2.type
    if type1 is long_type or type1.is_int:
        type1, type2 = (type2, type1)
    if type2 is long_type or type2.is_int:
        if type1.is_string or type1.is_ctuple:
            return True
        if self.is_builtin_seqmul_type(type1):
            return True
    return False