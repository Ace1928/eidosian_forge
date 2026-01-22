from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_link_libraries(ctx, tokens, breakstack):
    """
  ::

    link_libraries([item1 [item2 [...]]]
                   [[debug|optimized|general] <item>] ...)

  :see: https://cmake.org/cmake/help/latest/command/link_libraries.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, ['DEBUG', 'OPTIMIZED', 'GENERAL'], breakstack)