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
def visit_StatNode(self, node, is_listcontainer=False):
    stacktmp = self.is_in_statlist
    self.is_in_statlist = is_listcontainer
    self.visitchildren(node)
    self.is_in_statlist = stacktmp
    if not self.is_in_statlist and (not self.is_in_expr):
        return Nodes.StatListNode(pos=node.pos, stats=[node])
    else:
        return node