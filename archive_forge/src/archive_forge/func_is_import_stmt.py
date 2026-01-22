from lib2to3.fixer_util import (FromImport, Newline, is_import,
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms
from lib2to3.pygram import token
import re
def is_import_stmt(node):
    return node.type == syms.simple_stmt and node.children and is_import(node.children[0])