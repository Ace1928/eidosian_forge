import logging
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import NodeType, TreeNode
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.util import (
def parse_add_library(ctx, tokens, breakstack):
    """
  ``add_library()`` has several forms:

  * normal libraires
  * imported libraries
  * object libraries
  * alias libraries
  * interface libraries

  This function is just the dispatcher

  :see: https://cmake.org/cmake/help/latest/command/add_library.html
  """
    semantic_iter = iter_semantic_tokens(tokens)
    _ = next(semantic_iter, None)
    second_token = next(semantic_iter, None)
    if second_token is None:
        logger.warning('Invalid add_library() command at %s', tokens[0].get_location())
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    descriminator = second_token.spelling.upper()
    parsemap = {'OBJECT': parse_add_library_object, 'ALIAS': parse_add_library_alias, 'INTERFACE': parse_add_library_interface, 'IMPORTED': parse_add_library_imported}
    if descriminator in parsemap:
        return parsemap[descriminator](ctx, tokens, breakstack)
    third_token = next(semantic_iter, None)
    if third_token is not None:
        descriminator = third_token.spelling.upper()
        if descriminator == 'IMPORTED':
            return parse_add_library_imported(ctx, tokens, breakstack)
    sortable = True
    if '${' in second_token.spelling or '${' in third_token.spelling:
        sortable = False
    return parse_add_library_standard(ctx, tokens, breakstack, sortable)