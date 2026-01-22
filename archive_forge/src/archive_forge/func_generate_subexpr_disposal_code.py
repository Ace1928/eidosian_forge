from __future__ import absolute_import
from . import Nodes
from . import ExprNodes
from .Nodes import Node
from .ExprNodes import AtomicExprNode
from .PyrexTypes import c_ptr_type, c_bint_type
def generate_subexpr_disposal_code(self, code):
    self.subexpression.generate_disposal_code(code)