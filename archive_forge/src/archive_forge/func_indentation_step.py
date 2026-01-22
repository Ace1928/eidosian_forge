from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def indentation_step(node):
    """
    Dirty little trick to get the difference between each indentation level
    Implemented by finding the shortest indentation string
    (technically, the "least" of all of the indentation strings, but
    tabs and spaces mixed won't get this far, so those are synonymous.)
    """
    r = find_root(node)
    all_indents = set((i.value for i in r.pre_order() if i.type == token.INDENT))
    if not all_indents:
        return u'    '
    else:
        return min(all_indents)