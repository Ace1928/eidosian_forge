import logging
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (
def parse_add_library_alias(ctx, tokens, breakstack):
    """
  ::
    add_library(<name> ALIAS <target>)

    :see: https://cmake.org/cmake/help/latest/command/add_library.html#alias-libraries
  """
    return StandardArgTree.parse(ctx, tokens, npargs=3, kwargs={}, flags=['ALIAS'], breakstack=breakstack)