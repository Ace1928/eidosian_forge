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
def make_length_call():
    builtin_len = ExprNodes.NameNode(node.pos, name='len', entry=Builtin.builtin_scope.lookup('len'))
    return ExprNodes.SimpleCallNode(node.pos, function=builtin_len, args=[unpack_temp_node])