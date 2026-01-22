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
def serialize_modulenode_as_function(self, node):
    """
        Serialize the module-level code as a function so the debugger will know
        it's a "relevant frame" and it will know where to set the breakpoint
        for 'break modulename'.
        """
    self._serialize_modulenode_as_function(node, dict(name=node.full_module_name.rpartition('.')[-1], cname=node.module_init_func_cname(), pf_cname='', qualified_name='', lineno='1', is_initmodule_function='True'))