from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_install_targets(ctx, tokens, breakstack):
    """
  ::

    install_targets(<dir> [RUNTIME_DIRECTORY dir] target target)

  :see: https://cmake.org/cmake/help/latest/command/install_targets.html
  """
    kwargs = {'RUNTIME_DIRECTORY': PositionalParser(1)}
    return StandardArgTree.parse(ctx, tokens, '2+', kwargs, [], breakstack)