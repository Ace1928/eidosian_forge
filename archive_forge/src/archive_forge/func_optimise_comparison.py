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
def optimise_comparison(self, operand1, env, result_is_bool=False):
    if self.find_special_bool_compare_function(env, operand1, result_is_bool):
        self.is_pycmp = False
        self.type = PyrexTypes.c_bint_type
        if not operand1.type.is_pyobject:
            operand1 = operand1.coerce_to_pyobject(env)
    if self.cascade:
        operand2 = self.cascade.optimise_comparison(self.operand2, env, result_is_bool)
        if operand2 is not self.operand2:
            self.coerced_operand2 = operand2
    return operand1