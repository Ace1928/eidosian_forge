from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def mark_assignment(self, lhs, rhs=None, rhs_scope=None):
    if not self.flow.block:
        return
    if self.flow.exceptions:
        exc_descr = self.flow.exceptions[-1]
        self.flow.block.add_child(exc_descr.entry_point)
        self.flow.nextblock()
    if not rhs:
        rhs = self.object_expr
    if lhs.is_name:
        if lhs.entry is not None:
            entry = lhs.entry
        else:
            entry = self.env.lookup(lhs.name)
        if entry is None:
            return
        self.flow.mark_assignment(lhs, rhs, entry, rhs_scope=rhs_scope)
    elif lhs.is_sequence_constructor:
        for i, arg in enumerate(lhs.args):
            if arg.is_starred:
                item_node = TypedExprNode(Builtin.list_type, may_be_none=False, pos=arg.pos)
            elif rhs is self.object_expr:
                item_node = rhs
            else:
                item_node = rhs.inferable_item_node(i)
            self.mark_assignment(arg, item_node)
    else:
        self._visit(lhs)
    if self.flow.exceptions:
        exc_descr = self.flow.exceptions[-1]
        self.flow.block.add_child(exc_descr.entry_point)
        self.flow.nextblock()