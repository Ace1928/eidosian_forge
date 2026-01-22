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
def visit_with_directives(self, node, directives, contents_directives):
    if not directives:
        assert not contents_directives
        return self.visit_Node(node)
    old_directives = self.directives
    new_directives = Options.copy_inherited_directives(old_directives, **directives)
    if contents_directives is not None:
        new_contents_directives = Options.copy_inherited_directives(old_directives, **contents_directives)
    else:
        new_contents_directives = new_directives
    if new_directives == old_directives:
        return self.visit_Node(node)
    self.directives = new_directives
    if contents_directives is not None and new_contents_directives != new_directives:
        node.body = Nodes.StatListNode(node.body.pos, stats=[Nodes.CompilerDirectivesNode(node.body.pos, directives=new_contents_directives, body=node.body)])
    retbody = self.visit_Node(node)
    self.directives = old_directives
    if not isinstance(retbody, Nodes.StatListNode):
        retbody = Nodes.StatListNode(node.pos, stats=[retbody])
    return Nodes.CompilerDirectivesNode(pos=retbody.pos, body=retbody, directives=new_directives)