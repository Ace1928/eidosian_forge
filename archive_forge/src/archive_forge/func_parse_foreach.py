import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker, NodeType
from cmakelang.parse.common import TreeNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_foreach(ctx, tokens, breakstack):
    """
  ``parse_foreach()`` has a couple of forms:

  * the usual form
  * range form
  * in (lists/items) form

  This function is just the dispatcher

  :see: https://cmake.org/cmake/help/latest/command/foreach.html
  """
    semantic_iter = iter_semantic_tokens(tokens)
    _ = next(semantic_iter, None)
    second_token = next(semantic_iter, None)
    if second_token is None:
        logger.warning('Invalid foreach() statement at %s', tokens[0].get_location())
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    descriminator = second_token.spelling.upper()
    dispatchee = {'RANGE': parse_foreach_range, 'IN': parse_foreach_in}.get(descriminator, None)
    if dispatchee is not None:
        return dispatchee(ctx, tokens, breakstack)
    return parse_foreach_standard(ctx, tokens, breakstack)