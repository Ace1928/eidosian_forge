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
def get_slice_config(self):
    has_c_start, c_start, py_start = (False, '0', 'NULL')
    if self.start:
        has_c_start = not self.start.type.is_pyobject
        if has_c_start:
            c_start = self.start.result()
        else:
            py_start = '&%s' % self.start.py_result()
    has_c_stop, c_stop, py_stop = (False, '0', 'NULL')
    if self.stop:
        has_c_stop = not self.stop.type.is_pyobject
        if has_c_stop:
            c_stop = self.stop.result()
        else:
            py_stop = '&%s' % self.stop.py_result()
    py_slice = self.slice and '&%s' % self.slice.py_result() or 'NULL'
    return (has_c_start, has_c_stop, c_start, c_stop, py_start, py_stop, py_slice)