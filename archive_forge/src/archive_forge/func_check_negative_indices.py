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
def check_negative_indices(*nodes):
    """
    Raise a warning on nodes that are known to have negative numeric values.
    Used to find (potential) bugs inside of "wraparound=False" sections.
    """
    for node in nodes:
        if node is None or (not isinstance(node.constant_result, _py_int_types) and (not isinstance(node.constant_result, float))):
            continue
        if node.constant_result < 0:
            warning(node.pos, "the result of using negative indices inside of code sections marked as 'wraparound=False' is undefined", level=1)