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
def repr_of(self, node):
    if node is None:
        return '(none)'
    else:
        result = node.__class__.__name__
        if isinstance(node, ExprNodes.NameNode):
            result += '(type=%s, name="%s")' % (repr(node.type), node.name)
        elif isinstance(node, Nodes.DefNode):
            result += '(name="%s")' % node.name
        elif isinstance(node, Nodes.CFuncDefNode):
            result += '(name="%s")' % node.declared_name()
        elif isinstance(node, ExprNodes.AttributeNode):
            result += '(type=%s, attribute="%s")' % (repr(node.type), node.attribute)
        elif isinstance(node, (ExprNodes.ConstNode, ExprNodes.PyConstNode)):
            result += '(type=%s, value=%r)' % (repr(node.type), node.value)
        elif isinstance(node, ExprNodes.ExprNode):
            t = node.type
            result += '(type=%s)' % repr(t)
        elif node.pos:
            pos = node.pos
            path = pos[0].get_description()
            if '/' in path:
                path = path.split('/')[-1]
            if '\\' in path:
                path = path.split('\\')[-1]
            result += '(pos=(%s:%s:%s))' % (path, pos[1], pos[2])
        return result