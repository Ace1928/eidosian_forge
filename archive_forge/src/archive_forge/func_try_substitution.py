from __future__ import absolute_import
import re
from io import StringIO
from .Scanning import PyrexScanner, StringSourceDescriptor
from .Symtab import ModuleScope
from . import PyrexTypes
from .Visitor import VisitorTransform
from .Nodes import Node, StatListNode
from .ExprNodes import NameNode
from .StringEncoding import _unicode
from . import Parsing
from . import Main
from . import UtilNodes
def try_substitution(self, node, key):
    sub = self.substitutions.get(key)
    if sub is not None:
        pos = self.pos
        if pos is None:
            pos = node.pos
        return ApplyPositionAndCopy(pos)(sub)
    else:
        return self.visit_Node(node)