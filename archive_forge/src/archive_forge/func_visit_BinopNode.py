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
def visit_BinopNode(self, node):
    self._calculate_const(node)
    if node.constant_result is ExprNodes.not_a_constant:
        return node
    if isinstance(node.constant_result, float):
        return node
    operand1, operand2 = (node.operand1, node.operand2)
    if not operand1.is_literal or not operand2.is_literal:
        return node
    try:
        type1, type2 = (operand1.type, operand2.type)
        if type1 is None or type2 is None:
            return node
    except AttributeError:
        return node
    if type1.is_numeric and type2.is_numeric:
        widest_type = PyrexTypes.widest_numeric_type(type1, type2)
    else:
        widest_type = PyrexTypes.py_object_type
    target_class = self._widest_node_class(operand1, operand2)
    if target_class is None:
        return node
    elif target_class is ExprNodes.BoolNode and node.operator in '+-//<<%**>>':
        target_class = ExprNodes.IntNode
    elif target_class is ExprNodes.CharNode and node.operator in '+-//<<%**>>&|^':
        target_class = ExprNodes.IntNode
    if target_class is ExprNodes.IntNode:
        unsigned = getattr(operand1, 'unsigned', '') and getattr(operand2, 'unsigned', '')
        longness = 'LL'[:max(len(getattr(operand1, 'longness', '')), len(getattr(operand2, 'longness', '')))]
        value = hex(int(node.constant_result))
        value = Utils.strip_py2_long_suffix(value)
        new_node = ExprNodes.IntNode(pos=node.pos, unsigned=unsigned, longness=longness, value=value, constant_result=int(node.constant_result))
        if widest_type.is_pyobject or new_node.type.is_pyobject:
            new_node.type = PyrexTypes.py_object_type
        else:
            new_node.type = PyrexTypes.widest_numeric_type(widest_type, new_node.type)
    else:
        if target_class is ExprNodes.BoolNode:
            node_value = node.constant_result
        else:
            node_value = str(node.constant_result)
        new_node = target_class(pos=node.pos, type=widest_type, value=node_value, constant_result=node.constant_result)
    return new_node