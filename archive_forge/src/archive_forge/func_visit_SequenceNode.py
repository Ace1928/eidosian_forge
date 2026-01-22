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
def visit_SequenceNode(self, node):
    """Unpack *args in place if we can."""
    self.visitchildren(node)
    args = []
    for arg in node.args:
        if not arg.is_starred:
            args.append(arg)
        elif arg.target.is_sequence_constructor and (not arg.target.mult_factor):
            args.extend(arg.target.args)
        else:
            args.append(arg)
    node.args[:] = args
    self._calculate_const(node)
    return node