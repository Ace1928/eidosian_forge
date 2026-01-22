from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def nextblock(self, parent=None):
    """Create block children block linked to current or `parent` if given.

           NOTE: Block is added to self.blocks
        """
    block = ControlBlock()
    self.blocks.add(block)
    if parent:
        parent.add_child(block)
    elif self.block:
        self.block.add_child(block)
    self.block = block
    return self.block