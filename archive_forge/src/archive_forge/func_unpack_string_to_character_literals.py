from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def unpack_string_to_character_literals(literal):
    chars = []
    pos = literal.pos
    stype = literal.__class__
    sval = literal.value
    sval_type = sval.__class__
    for char in sval:
        cval = sval_type(char)
        chars.append(stype(pos, value=cval, constant_result=cval))
    return chars