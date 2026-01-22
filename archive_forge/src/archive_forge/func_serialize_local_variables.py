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
def serialize_local_variables(self, entries):
    for entry in entries.values():
        if not entry.cname:
            continue
        if entry.type.is_pyobject:
            vartype = 'PythonObject'
        else:
            vartype = 'CObject'
        if entry.from_closure:
            cname = '%s->%s' % (Naming.cur_scope_cname, entry.outer_entry.cname)
            qname = '%s.%s.%s' % (entry.scope.outer_scope.qualified_name, entry.scope.name, entry.name)
        elif entry.in_closure:
            cname = '%s->%s' % (Naming.cur_scope_cname, entry.cname)
            qname = entry.qualified_name
        else:
            cname = entry.cname
            qname = entry.qualified_name
        if not entry.pos:
            lineno = '0'
        else:
            lineno = str(entry.pos[1])
        attrs = dict(name=entry.name, cname=cname, qualified_name=qname, type=vartype, lineno=lineno)
        self.tb.start('LocalVar', attrs)
        self.tb.end('LocalVar')