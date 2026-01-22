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
def merged_indices(self, indices):
    """Return a new list of indices/slices with 'indices' merged into the current ones
        according to slicing rules.
        Is used to implement "view[i][j]" => "view[i, j]".
        Return None if the indices cannot (easily) be merged at compile time.
        """
    if not indices:
        return None
    new_indices = self.original_indices[:]
    indices = indices[:]
    for i, s in enumerate(self.original_indices):
        if s.is_slice:
            if s.start.is_none and s.stop.is_none and s.step.is_none:
                new_indices[i] = indices[0]
                indices.pop(0)
                if not indices:
                    return new_indices
            else:
                return None
        elif not s.type.is_int:
            return None
    if indices:
        if len(new_indices) + len(indices) > self.base.type.ndim:
            return None
        new_indices += indices
    return new_indices