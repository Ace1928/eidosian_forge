from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def visit_ClassDefNode(self, node):
    oldname = self.class_name
    oldclass = self.class_node
    self.class_node = node
    try:
        self.class_name = node.name
    except AttributeError:
        self.class_name = node.class_name
    self.visitchildren(node)
    self.class_name = oldname
    self.class_node = oldclass
    return node