import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_write(ctx, tokens, breakstack):
    """
  ::

    file(WRITE <filename> <content>...)
    file(APPEND <filename> <content>...)

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#writing
  """
    tree = ArgGroupNode()
    consume_whitespace_and_comments(ctx, tokens, tree)
    tree.children.append(PositionalGroupNode.parse(ctx, tokens, 2, ['WRITE', 'APPEND'], breakstack))
    consume_whitespace_and_comments(ctx, tokens, tree)
    tree.children.append(PositionalGroupNode.parse(ctx, tokens, '+', [], breakstack))
    return tree