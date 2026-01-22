from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_target_sources(ctx, tokens, breakstack):
    """
  ::

    target_sources(<target>
      <INTERFACE|PUBLIC|PRIVATE> [items1...]
      [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

  :see: https://cmake.org/cmake/help/latest/command/target_sources.html
  """
    kwargs = {'INTERFACE': PositionalParser('+'), 'PUBLIC': PositionalParser('+'), 'PRIVATE': PositionalParser('+')}
    return StandardArgTree.parse(ctx, tokens, 1, kwargs, [], breakstack)