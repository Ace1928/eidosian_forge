import logging
from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (
def parse_add_executable(ctx, tokens, breakstack):
    """
  ``add_executable()`` has a couple of forms:

  * normal executables
  * imported executables
  * alias executables

  This function is just the dispatcher

  :see: https://cmake.org/cmake/help/latest/command/add_executable.html
  """
    semantic_iter = iter_semantic_tokens(tokens)
    _ = next(semantic_iter, None)
    second_token = next(semantic_iter, None)
    if second_token is None:
        logger.warning('Invalid add_executable() command at %s', tokens[0].get_location())
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    descriminator = second_token.spelling.upper()
    parsemap = {'ALIAS': parse_add_executable_alias, 'IMPORTED': parse_add_executable_imported}
    if descriminator in parsemap:
        return parsemap[descriminator](ctx, tokens, breakstack)
    sortable = True
    if '${' in second_token.spelling:
        sortable = False
    return parse_add_executable_standard(ctx, tokens, breakstack, sortable)