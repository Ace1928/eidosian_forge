from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def import_binding_scope(node):
    """
    Generator yields all nodes for which a node (an import_stmt) has scope
    The purpose of this is for a call to _find() on each of them
    """
    assert node.type in _import_stmts
    test = node.next_sibling
    while test.type == token.SEMI:
        nxt = test.next_sibling
        if nxt.type == token.NEWLINE:
            break
        else:
            yield nxt
        test = nxt.next_sibling
    parent = node.parent
    assert parent.type == syms.simple_stmt
    test = parent.next_sibling
    while test is not None:
        yield test
        test = test.next_sibling
    context = parent.parent
    if context.type in _compound_stmts:
        c = context
        while c.next_sibling is not None:
            yield c.next_sibling
            c = c.next_sibling
        context = context.parent
    p = context.parent
    if p is None:
        return
    while p.type in _compound_stmts:
        if context.type == syms.suite:
            yield context
        context = context.next_sibling
        if context is None:
            context = p.parent
            p = context.parent
            if p is None:
                break