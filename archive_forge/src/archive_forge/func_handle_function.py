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
def handle_function(self, node):
    if not getattr(node, 'decorators', None):
        return self.visit_Node(node)
    for i, decorator in enumerate(node.decorators):
        decorator = decorator.decorator
        if isinstance(decorator, ExprNodes.CallNode) and decorator.function.is_name and (decorator.function.name == 'cname'):
            args, kwargs = decorator.explicit_args_kwds()
            if kwargs:
                raise AssertionError('cname decorator does not take keyword arguments')
            if len(args) != 1:
                raise AssertionError('cname decorator takes exactly one argument')
            if not (args[0].is_literal and args[0].type == Builtin.str_type):
                raise AssertionError('argument to cname decorator must be a string literal')
            cname = args[0].compile_time_value(None)
            del node.decorators[i]
            node = Nodes.CnameDecoratorNode(pos=node.pos, node=node, cname=cname)
            break
    return self.visit_Node(node)