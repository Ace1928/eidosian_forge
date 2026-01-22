from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def touch_import_top(package, name_to_import, node):
    """Works like `does_tree_import` but adds an import statement at the
    top if it was not imported (but below any __future__ imports) and below any
    comments such as shebang lines).

    Based on lib2to3.fixer_util.touch_import()

    Calling this multiple times adds the imports in reverse order.

    Also adds "standard_library.install_aliases()" after "from future import
    standard_library".  This should probably be factored into another function.
    """
    root = find_root(node)
    if does_tree_import(package, name_to_import, root):
        return
    found = False
    for name in ['absolute_import', 'division', 'print_function', 'unicode_literals']:
        if does_tree_import('__future__', name, root):
            found = True
            break
    if found:
        start, end = (None, None)
        for idx, node in enumerate(root.children):
            if check_future_import(node):
                start = idx
                idx2 = start
                while node:
                    node = node.next_sibling
                    idx2 += 1
                    if not check_future_import(node):
                        end = idx2
                        break
                break
        assert start is not None
        assert end is not None
        insert_pos = end
    else:
        for idx, node in enumerate(root.children):
            if node.type != syms.simple_stmt:
                break
            if not is_docstring(node):
                break
        insert_pos = idx
    children_hooks = []
    if package is None:
        import_ = Node(syms.import_name, [Leaf(token.NAME, u'import'), Leaf(token.NAME, name_to_import, prefix=u' ')])
    else:
        import_ = FromImport(package, [Leaf(token.NAME, name_to_import, prefix=u' ')])
        if name_to_import == u'standard_library':
            install_hooks = Node(syms.simple_stmt, [Node(syms.power, [Leaf(token.NAME, u'standard_library'), Node(syms.trailer, [Leaf(token.DOT, u'.'), Leaf(token.NAME, u'install_aliases')]), Node(syms.trailer, [Leaf(token.LPAR, u'('), Leaf(token.RPAR, u')')])])])
            children_hooks = [install_hooks, Newline()]
    children_import = [import_, Newline()]
    old_prefix = root.children[insert_pos].prefix
    root.children[insert_pos].prefix = u''
    root.insert_child(insert_pos, Node(syms.simple_stmt, children_import, prefix=old_prefix))
    if len(children_hooks) > 0:
        root.insert_child(insert_pos + 1, Node(syms.simple_stmt, children_hooks))