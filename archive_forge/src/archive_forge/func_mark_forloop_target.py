from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def mark_forloop_target(self, node):
    is_special = False
    sequence = node.iterator.sequence
    target = node.target
    env = node.iterator.expr_scope or self.env
    if isinstance(sequence, ExprNodes.SimpleCallNode):
        function = sequence.function
        if sequence.self is None and function.is_name:
            entry = env.lookup(function.name)
            if not entry or entry.is_builtin:
                if function.name == 'reversed' and len(sequence.args) == 1:
                    sequence = sequence.args[0]
                elif function.name == 'enumerate' and len(sequence.args) == 1:
                    if target.is_sequence_constructor and len(target.args) == 2:
                        iterator = sequence.args[0]
                        if iterator.is_name:
                            iterator_type = iterator.infer_type(env)
                            if iterator_type.is_builtin_type:
                                self.mark_assignment(target.args[0], ExprNodes.IntNode(target.pos, value='PY_SSIZE_T_MAX', type=PyrexTypes.c_py_ssize_t_type), rhs_scope=node.iterator.expr_scope)
                                target = target.args[1]
                                sequence = sequence.args[0]
    if isinstance(sequence, ExprNodes.SimpleCallNode):
        function = sequence.function
        if sequence.self is None and function.is_name:
            entry = env.lookup(function.name)
            if not entry or entry.is_builtin:
                if function.name in ('range', 'xrange'):
                    is_special = True
                    for arg in sequence.args[:2]:
                        self.mark_assignment(target, arg, rhs_scope=node.iterator.expr_scope)
                    if len(sequence.args) > 2:
                        self.mark_assignment(target, self.constant_folder(ExprNodes.binop_node(node.pos, '+', sequence.args[0], sequence.args[2])), rhs_scope=node.iterator.expr_scope)
    if not is_special:
        self.mark_assignment(target, node.item, rhs_scope=node.iterator.expr_scope)