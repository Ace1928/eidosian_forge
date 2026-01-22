from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def mark_deletion(self, node, entry):
    if self.block and self.is_tracked(entry):
        assignment = NameDeletion(node, entry)
        self.block.stats.append(assignment)
        self.block.gen[entry] = Uninitialized
        self.entries.add(entry)