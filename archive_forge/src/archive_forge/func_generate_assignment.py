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
def generate_assignment(self, node, name, value):
    entry = node.scope.lookup_here(name)
    lhs = ExprNodes.NameNode(node.pos, name=EncodedString(name), entry=entry)
    rhs = ExprNodes.StringNode(node.pos, value=value.as_utf8_string(), unicode_value=value)
    node.body.stats.insert(0, Nodes.SingleAssignmentNode(node.pos, lhs=lhs, rhs=rhs).analyse_expressions(self.current_env()))