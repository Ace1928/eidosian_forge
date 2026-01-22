from lib2to3 import fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import syms
def param_without_annotations(node):
    return node.children[0]