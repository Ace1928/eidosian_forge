from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
def visitchild(self, parent, attr, idx=0):
    child = getattr(parent, attr)
    if child is not None:
        node = self._visitchild(child, parent, attr, idx)
        if node is not child:
            setattr(parent, attr, node)
        child = node
    return child