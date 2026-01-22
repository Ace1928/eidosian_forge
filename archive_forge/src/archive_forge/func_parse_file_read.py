import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_read(ctx, tokens, breakstack):
    """
  ::

    file(READ <filename> <variable>
         [OFFSET <offset>] [LIMIT <max-in>] [HEX])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#read
  """
    return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'OFFSET': PositionalParser(1), 'LIMIT': PositionalParser(1)}, flags=['READ', 'HEX'], breakstack=breakstack)