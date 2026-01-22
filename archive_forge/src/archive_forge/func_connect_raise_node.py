import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def connect_raise_node(self, node, except_guards):
    """Adds extra connection between a raise node and containing except guards.

    The node is a graph node, not an ast node.

    Args:
      node: Node
      except_guards: Tuple[ast.AST, ...], the except sections that guard node
    """
    for guard in except_guards:
        if guard in self.raises:
            self.raises[guard].append(node)
        else:
            self.raises[guard] = [node]