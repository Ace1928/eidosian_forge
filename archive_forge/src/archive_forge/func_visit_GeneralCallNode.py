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
def visit_GeneralCallNode(self, node):
    function = node.function.as_cython_attribute()
    if function == u'cast':
        args = node.positional_args.args
        kwargs = node.keyword_args.compile_time_value(None)
        if len(args) != 2 or len(kwargs) > 1 or (len(kwargs) == 1 and 'typecheck' not in kwargs):
            error(node.function.pos, u'cast() takes exactly two arguments and an optional typecheck keyword')
        else:
            type = args[0].analyse_as_type(self.current_env())
            if type:
                typecheck = kwargs.get('typecheck', False)
                node = ExprNodes.TypecastNode(node.function.pos, type=type, operand=args[1], typecheck=typecheck)
            else:
                error(args[0].pos, 'Not a type')
    self.visitchildren(node)
    return node