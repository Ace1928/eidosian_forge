from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_unset(ctx, tokens, breakstack):
    """
  ::

    unset(<variable> [CACHE | PARENT_SCOPE])
    unset(ENV{<variable>})

  :see: https://cmake.org/cmake/help/latest/command/unset.html
  """
    flags = ['CACHE', 'PARENT_SCOPE']
    return StandardArgTree.parse(ctx, tokens, '+', {}, flags, breakstack)