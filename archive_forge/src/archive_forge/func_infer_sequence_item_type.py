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
def infer_sequence_item_type(env, seq_node, index_node=None, seq_type=None):
    if not seq_node.is_sequence_constructor:
        if seq_type is None:
            seq_type = seq_node.infer_type(env)
        if seq_type is tuple_type:
            if seq_node.cf_state and len(seq_node.cf_state) == 1:
                try:
                    seq_node = seq_node.cf_state[0].rhs
                except AttributeError:
                    pass
    if seq_node is not None and seq_node.is_sequence_constructor:
        if index_node is not None and index_node.has_constant_result():
            try:
                item = seq_node.args[index_node.constant_result]
            except (ValueError, TypeError, IndexError):
                pass
            else:
                return item.infer_type(env)
        item_types = {item.infer_type(env) for item in seq_node.args}
        if len(item_types) == 1:
            return item_types.pop()
    return None