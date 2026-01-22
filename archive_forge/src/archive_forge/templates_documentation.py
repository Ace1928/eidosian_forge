import ast
import textwrap
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
Prepares a replacement AST that's safe to swap in for a node.

    Args:
      replaced: ast.AST, the node being replaced
      key: Hashable, the key of the replacement AST
    Returns:
      ast.AST, the replacement AST
    