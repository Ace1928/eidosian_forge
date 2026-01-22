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
def handle_bufferdefaults(self, decl):
    if not isinstance(decl.default, ExprNodes.DictNode):
        raise PostParseError(decl.pos, ERR_BUF_DEFAULTS)
    self.scope_node.buffer_defaults_node = decl.default
    self.scope_node.buffer_defaults_pos = decl.pos