from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def suitify(parent):
    """
    Turn the stuff after the first colon in parent's children
    into a suite, if it wasn't already
    """
    for node in parent.children:
        if node.type == syms.suite:
            return
    for i, node in enumerate(parent.children):
        if node.type == token.COLON:
            break
    else:
        raise ValueError(u"No class suite and no ':'!")
    suite = Node(syms.suite, [Newline(), Leaf(token.INDENT, indentation(node) + indentation_step(node))])
    one_node = parent.children[i + 1]
    one_node.remove()
    one_node.prefix = u''
    suite.append_child(one_node)
    parent.append_child(suite)