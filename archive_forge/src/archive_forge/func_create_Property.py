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
def create_Property(self, entry):
    if entry.visibility == 'public':
        if entry.type.is_pyobject:
            template = self.basic_pyobject_property
        else:
            template = self.basic_property
    elif entry.visibility == 'readonly':
        template = self.basic_property_ro
    property = template.substitute({u'ATTR': ExprNodes.AttributeNode(pos=entry.pos, obj=ExprNodes.NameNode(pos=entry.pos, name='self'), attribute=entry.name)}, pos=entry.pos).stats[0]
    property.name = entry.name
    property.doc = entry.doc
    return property