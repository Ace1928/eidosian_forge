from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def visit_SimpleCallNode(self, node):
    function = node.function.as_cython_attribute()
    if function:
        if function in InterpretCompilerDirectives.unop_method_nodes:
            if len(node.args) != 1:
                error(node.function.pos, u'%s() takes exactly one argument' % function)
            else:
                node = InterpretCompilerDirectives.unop_method_nodes[function](node.function.pos, operand=node.args[0])
        elif function in InterpretCompilerDirectives.binop_method_nodes:
            if len(node.args) != 2:
                error(node.function.pos, u'%s() takes exactly two arguments' % function)
            else:
                node = InterpretCompilerDirectives.binop_method_nodes[function](node.function.pos, operand1=node.args[0], operand2=node.args[1])
        elif function == u'cast':
            if len(node.args) != 2:
                error(node.function.pos, u'cast() takes exactly two arguments and an optional typecheck keyword')
            else:
                type = node.args[0].analyse_as_type(self.current_env())
                if type:
                    node = ExprNodes.TypecastNode(node.function.pos, type=type, operand=node.args[1], typecheck=False)
                else:
                    error(node.args[0].pos, 'Not a type')
        elif function == u'sizeof':
            if len(node.args) != 1:
                error(node.function.pos, u'sizeof() takes exactly one argument')
            else:
                type = node.args[0].analyse_as_type(self.current_env())
                if type:
                    node = ExprNodes.SizeofTypeNode(node.function.pos, arg_type=type)
                else:
                    node = ExprNodes.SizeofVarNode(node.function.pos, operand=node.args[0])
        elif function == 'cmod':
            if len(node.args) != 2:
                error(node.function.pos, u'cmod() takes exactly two arguments')
            else:
                node = ExprNodes.binop_node(node.function.pos, '%', node.args[0], node.args[1])
                node.cdivision = True
        elif function == 'cdiv':
            if len(node.args) != 2:
                error(node.function.pos, u'cdiv() takes exactly two arguments')
            else:
                node = ExprNodes.binop_node(node.function.pos, '/', node.args[0], node.args[1])
                node.cdivision = True
        elif function == u'set':
            node.function = ExprNodes.NameNode(node.pos, name=EncodedString('set'))
        elif function == u'staticmethod':
            node.function = ExprNodes.NameNode(node.pos, name=EncodedString('staticmethod'))
        elif self.context.cython_scope.lookup_qualified_name(function):
            pass
        else:
            error(node.function.pos, u"'%s' not a valid cython language construct" % function)
    self.visitchildren(node)
    if isinstance(node, ExprNodes.SimpleCallNode) and node.function.is_name:
        func_name = node.function.name
        if func_name in ('dir', 'locals', 'vars'):
            return self._inject_locals(node, func_name)
        if func_name == 'eval':
            return self._inject_eval(node, func_name)
        if func_name == 'super':
            return self._inject_super(node, func_name)
    return node