from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
def fixup_indent(suite):
    """ If an INDENT is followed by a thing with a prefix then nuke the prefix
        Otherwise we get in trouble when removing __metaclass__ at suite start
    """
    kids = suite.children[::-1]
    while kids:
        node = kids.pop()
        if node.type == token.INDENT:
            break
    while kids:
        node = kids.pop()
        if isinstance(node, Leaf) and node.type != token.DEDENT:
            if node.prefix:
                node.prefix = u''
            return
        else:
            kids.extend(node.children[::-1])