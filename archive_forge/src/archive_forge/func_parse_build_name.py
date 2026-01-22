from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_build_name(ctx, tokens, breakstack):
    """
  ::

    build_name(variable)

  :see: https://cmake.org/cmake/help/latest/command/build_name.html
  """
    return StandardArgTree.parse(ctx, tokens, 1, {}, [], breakstack)