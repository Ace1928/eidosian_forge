from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_link_directories(ctx, tokens, breakstack):
    """
  ::

    link_directories([AFTER|BEFORE] directory1 [directory2 ...])

  :see: https://cmake.org/cmake/help/latest/command/link_directories.html
  """
    return StandardArgTree.parse(ctx, tokens, '+', {}, ['AFTER', 'BEFORE'], breakstack)