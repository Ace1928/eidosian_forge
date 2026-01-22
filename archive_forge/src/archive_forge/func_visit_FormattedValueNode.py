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
def visit_FormattedValueNode(self, node):
    self.visitchildren(node)
    conversion_char = node.conversion_char or 's'
    if isinstance(node.format_spec, ExprNodes.UnicodeNode) and (not node.format_spec.value):
        node.format_spec = None
    if node.format_spec is None and isinstance(node.value, ExprNodes.IntNode):
        value = EncodedString(node.value.value)
        if value.isdigit():
            return ExprNodes.UnicodeNode(node.value.pos, value=value, constant_result=value)
    if node.format_spec is None and conversion_char == 's':
        value = None
        if isinstance(node.value, ExprNodes.UnicodeNode):
            value = node.value.value
        elif isinstance(node.value, ExprNodes.StringNode):
            value = node.value.unicode_value
        if value is not None:
            return ExprNodes.UnicodeNode(node.value.pos, value=value, constant_result=value)
    return node