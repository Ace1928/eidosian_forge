import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file_copy(ctx, tokens, breakstack):
    """
  ::

    file(<COPY|INSTALL> <files>... DESTINATION <dir>
         [FILE_PERMISSIONS <permissions>...]
         [DIRECTORY_PERMISSIONS <permissions>...]
         [NO_SOURCE_PERMISSIONS] [USE_SOURCE_PERMISSIONS]
         [FILES_MATCHING]
         [[PATTERN <pattern> | REGEX <regex>]
          [EXCLUDE] [PERMISSIONS <permissions>...]] [...])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html#filesystem
  """
    return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'COPY': PositionalParser('*'), 'DESTINATION': PositionalParser(1), 'FILE_PERMISSIONS': PositionalParser('+', flags=['OWNER_READ', 'OWNER_WRITE', 'OWNER_EXECUTE', 'GROUP_READ', 'GROUP_WRITE', 'GROUP_EXECUTE', 'WORLD_READ', 'WORLD_WRITE', 'WORLD_EXECUTE', 'SETUID', 'SETGID']), 'DIRECTORY_PERMISSIONS': PositionalParser('+'), 'PATTERN': PatternNode.parse, 'REGEX': PatternNode.parse}, flags=['INSTALL', 'NO_SOURCE_PERMISSIONS', 'USE_SOURCE_PERMISSIONS', 'FILES_MATCHING'], breakstack=breakstack)