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
def value_as_c_integer_string(self):
    value = self.value
    if len(value) <= 2:
        return value
    neg_sign = ''
    if value[0] == '-':
        neg_sign = '-'
        value = value[1:]
    if value[0] == '0':
        literal_type = value[1]
        if neg_sign and literal_type in 'oOxX0123456789' and value[2:].isdigit():
            value = str(Utils.str_to_number(value))
        elif literal_type in 'oO':
            value = '0' + value[2:]
        elif literal_type in 'bB':
            value = str(int(value[2:], 2))
    elif value.isdigit() and (not self.unsigned) and (not self.longness):
        if not neg_sign:
            value = '0x%X' % int(value)
    return neg_sign + value