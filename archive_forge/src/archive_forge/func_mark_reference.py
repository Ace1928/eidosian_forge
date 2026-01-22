from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def mark_reference(self, node, entry):
    if self.block and self.is_tracked(entry):
        self.block.stats.append(NameReference(node, entry))
        self.entries.add(entry)