import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def can_ignore(self, node):
    """Returns True if the node can safely be assumed not to touch variables."""
    ast_node = node.ast_node
    if anno.hasanno(ast_node, anno.Basic.SKIP_PROCESSING):
        return True
    return isinstance(ast_node, (gast.Break, gast.Continue, gast.Raise, gast.Pass))