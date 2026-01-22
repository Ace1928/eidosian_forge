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
def visit_MergedSequenceNode(self, node):
    """Unpack *args in place if we can."""
    self.visitchildren(node)
    is_set = node.type is Builtin.set_type
    args = []
    values = []

    def add(arg):
        if is_set and arg.is_set_literal or (arg.is_sequence_constructor and (not arg.mult_factor)):
            if values:
                values[0].args.extend(arg.args)
            else:
                values.append(arg)
        elif isinstance(arg, ExprNodes.MergedSequenceNode):
            for child_arg in arg.args:
                add(child_arg)
        else:
            if values:
                args.append(values[0])
                del values[:]
            args.append(arg)
    for arg in node.args:
        add(arg)
    if values:
        args.append(values[0])
    if len(args) == 1:
        arg = args[0]
        if is_set and arg.is_set_literal or (arg.is_sequence_constructor and arg.type is node.type) or isinstance(arg, ExprNodes.MergedSequenceNode):
            return arg
    node.args[:] = args
    self._calculate_const(node)
    return node