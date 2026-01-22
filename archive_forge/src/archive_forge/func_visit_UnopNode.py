from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def visit_UnopNode(self, node):
    self._calculate_const(node)
    if not node.has_constant_result():
        if node.operator == '!':
            return self._handle_NotNode(node)
        return node
    if not node.operand.is_literal:
        return node
    if node.operator == '!':
        return self._bool_node(node, node.constant_result)
    elif isinstance(node.operand, ExprNodes.BoolNode):
        return ExprNodes.IntNode(node.pos, value=str(int(node.constant_result)), type=PyrexTypes.c_int_type, constant_result=int(node.constant_result))
    elif node.operator == '+':
        return self._handle_UnaryPlusNode(node)
    elif node.operator == '-':
        return self._handle_UnaryMinusNode(node)
    return node