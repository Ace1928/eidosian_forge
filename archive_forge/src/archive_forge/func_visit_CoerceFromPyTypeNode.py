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
def visit_CoerceFromPyTypeNode(self, node):
    """Drop redundant conversion nodes after tree changes.

        Also, optimise away calls to Python's builtin int() and
        float() if the result is going to be coerced back into a C
        type anyway.
        """
    self.visitchildren(node)
    arg = node.arg
    if not arg.type.is_pyobject:
        if node.type != arg.type:
            arg = arg.coerce_to(node.type, self.current_env())
        return arg
    if isinstance(arg, ExprNodes.PyTypeTestNode):
        arg = arg.arg
    if arg.is_literal:
        if node.type.is_int and isinstance(arg, ExprNodes.IntNode) or (node.type.is_float and isinstance(arg, ExprNodes.FloatNode)) or (node.type.is_int and isinstance(arg, ExprNodes.BoolNode)):
            return arg.coerce_to(node.type, self.current_env())
    elif isinstance(arg, ExprNodes.CoerceToPyTypeNode):
        if arg.type is PyrexTypes.py_object_type:
            if node.type.assignable_from(arg.arg.type):
                return arg.arg.coerce_to(node.type, self.current_env())
        elif arg.type is Builtin.unicode_type:
            if arg.arg.type.is_unicode_char and node.type.is_unicode_char:
                return arg.arg.coerce_to(node.type, self.current_env())
    elif isinstance(arg, ExprNodes.SimpleCallNode):
        if node.type.is_int or node.type.is_float:
            return self._optimise_numeric_cast_call(node, arg)
    elif arg.is_subscript:
        index_node = arg.index
        if isinstance(index_node, ExprNodes.CoerceToPyTypeNode):
            index_node = index_node.arg
        if index_node.type.is_int:
            return self._optimise_int_indexing(node, arg, index_node)
    return node