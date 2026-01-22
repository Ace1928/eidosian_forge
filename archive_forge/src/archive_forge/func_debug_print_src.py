import collections
import enum
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import templates
def debug_print_src(self, node):
    """Helper method useful for debugging. Prints the AST as code."""
    if __debug__:
        print(parser.unparse(node))
    return node