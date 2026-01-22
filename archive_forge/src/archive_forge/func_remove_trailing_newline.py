from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
def remove_trailing_newline(node):
    if node.children and node.children[-1].type == token.NEWLINE:
        node.children[-1].remove()