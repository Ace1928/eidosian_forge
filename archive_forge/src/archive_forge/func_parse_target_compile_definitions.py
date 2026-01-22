from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_target_compile_definitions(ctx, tokens, breakstack):
    """
  ::

    target_compile_definitions(<target>
        <INTERFACE|PUBLIC|PRIVATE> [items1...]
        [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])

  :see: https://cmake.org/cmake/help/latest/command/target_compile_definitions.html
  """
    kwargs = {'INTERFACE': PositionalParser('+'), 'PUBLIC': PositionalParser('+'), 'PRIVATE': PositionalParser('+')}
    return StandardArgTree.parse(ctx, tokens, 1, kwargs, [], breakstack)