import logging
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (
def parse_add_library_imported(ctx, tokens, breakstack):
    """
  ::
    add_library(<name> <SHARED|STATIC|MODULE|OBJECT|UNKNOWN> IMPORTED
                [GLOBAL])

    :see: https://cmake.org/cmake/help/latest/command/add_library.html
  """
    return StandardArgTree.parse(ctx, tokens, npargs='+', kwargs={}, flags=['SHARED', 'STATIC', 'MODULE', 'OBJECT', 'UNKOWN', 'IMPORTED', 'GLOBAL'], breakstack=breakstack)