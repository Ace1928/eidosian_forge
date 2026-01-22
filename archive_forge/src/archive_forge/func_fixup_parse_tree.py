from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
def fixup_parse_tree(cls_node):
    """ one-line classes don't get a suite in the parse tree so we add
        one to normalize the tree
    """
    for node in cls_node.children:
        if node.type == syms.suite:
            return
    for i, node in enumerate(cls_node.children):
        if node.type == token.COLON:
            break
    else:
        raise ValueError("No class suite and no ':'!")
    suite = Node(syms.suite, [])
    while cls_node.children[i + 1:]:
        move_node = cls_node.children[i + 1]
        suite.append_child(move_node.clone())
        move_node.remove()
    cls_node.append_child(suite)
    node = suite