from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_use_mangled_mesa(ctx, tokens, breakstack):
    """
  ::

    use_mangled_mesa(PATH_TO_MESA OUTPUT_DIRECTORY)

  :see: https://cmake.org/cmake/help/latest/command/use_mangled_mesa.html
  """
    return StandardArgTree.parse(ctx, tokens, 2, {}, [], breakstack)