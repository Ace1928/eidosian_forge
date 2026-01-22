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
def visit_scope(self, node, scope_type):
    prev = (self.scope_type, self.scope_node)
    self.scope_type = scope_type
    self.scope_node = node
    self._process_children(node)
    self.scope_type, self.scope_node = prev
    return node