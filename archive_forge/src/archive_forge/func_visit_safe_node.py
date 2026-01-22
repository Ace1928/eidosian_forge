from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def visit_safe_node(self, node):
    self.might_overflow, saved = (False, self.might_overflow)
    self.visitchildren(node)
    self.might_overflow = saved
    return node