from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_subdir_depends(ctx, tokens, breakstack):
    """
  ::

    subdir_depends(subdir dep1 dep2 ...)

  :see: https://cmake.org/cmake/help/latest/command/subdir_depends.html
  """
    return StandardArgTree.parse(ctx, tokens, '2+', {}, [], breakstack)