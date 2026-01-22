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
def find_entries_used_in_closures(self, node):
    from_closure = []
    in_closure = []
    for scope in node.local_scope.iter_local_scopes():
        for name, entry in scope.entries.items():
            if not name:
                continue
            if entry.from_closure:
                from_closure.append((name, entry))
            elif entry.in_closure:
                in_closure.append((name, entry))
    return (from_closure, in_closure)