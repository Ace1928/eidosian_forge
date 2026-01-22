from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_update(ctx, tokens, breakstack):
    """
  ::

    ctest_update([SOURCE <source-dir>]
                 [RETURN_VALUE <result-var>]
                 [CAPTURE_CMAKE_ERROR <result-var>]
                 [QUIET])

  :see: https://cmake.org/cmake/help/latest/command/ctest_update.html
  """
    kwargs = {'SOURCE': PositionalParser(1), 'RETURN_VALUE': PositionalParser(1), 'CAPTURE_CMAKE_ERROR': PositionalParser(1)}
    flags = ['QUIET']
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, flags, breakstack)