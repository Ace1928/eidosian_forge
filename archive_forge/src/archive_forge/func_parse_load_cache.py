from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_load_cache(ctx, tokens, breakstack):
    """
  ::

    load_cache(pathToBuildDirectory READ_WITH_PREFIX prefix entry1...)
    load_cache(pathToBuildDirectory [EXCLUDE entry1...]
               [INCLUDE_INTERNALS entry1...])

  :see: https://cmake.org/cmake/help/latest/command/load_cache.html
  """
    kwargs = {'READ_WITH_PREFIX': PositionalParser('2+'), 'EXCLUDE': PositionalParser('+'), 'INCLUDE_INTERNALS': PositionalParser('+')}
    return StandardArgTree.parse(ctx, tokens, 1, kwargs, [], breakstack)