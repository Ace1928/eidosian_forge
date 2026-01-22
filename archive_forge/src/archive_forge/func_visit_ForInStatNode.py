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
def visit_ForInStatNode(self, node):
    """Rewrite 'for i in cython.parallel.prange(...):'"""
    self.visitchild(node, 'iterator')
    self.visitchild(node, 'target')
    in_prange = isinstance(node.iterator.sequence, Nodes.ParallelRangeNode)
    previous_state = self.state
    if in_prange:
        parallel_range_node = node.iterator.sequence
        parallel_range_node.target = node.target
        parallel_range_node.body = node.body
        parallel_range_node.else_clause = node.else_clause
        node = parallel_range_node
        if not isinstance(node.target, ExprNodes.NameNode):
            error(node.target.pos, 'Can only iterate over an iteration variable')
        self.state = 'prange'
    self.visitchild(node, 'body')
    self.state = previous_state
    self.visitchild(node, 'else_clause')
    return node