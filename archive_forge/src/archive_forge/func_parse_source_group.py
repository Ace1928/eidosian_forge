from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_source_group(ctx, tokens, breakstack):
    """
  ::

    source_group(<name> [FILES <src>...] [REGULAR_EXPRESSION <regex>])
    source_group(TREE <root> [PREFIX <prefix>] [FILES <src>...])
    source_group(<name> <regex>)

  :see: https://cmake.org/cmake/help/latest/command/source_group.html
  """
    descriminator = get_nth_semantic_token(tokens, 0)
    if descriminator is not None and descriminator.spelling.upper() == 'TREE':
        kwargs = {'PREFIX': PositionalParser(1), 'FILES': PositionalParser('+')}
        return StandardArgTree.parse(ctx, tokens, 2, kwargs, [], breakstack)
    kwargs = {'FILES': PositionalParser('+'), 'REGULAR_EXPRESSION': PositionalParser(1)}
    descriminator = get_nth_semantic_token(tokens, 1)
    if descriminator is not None and descriminator.spelling.upper() not in kwargs:
        return StandardArgTree.parse(ctx, tokens, 2, {}, [], breakstack)
    return StandardArgTree.parse(ctx, tokens, 1, kwargs, [], breakstack)