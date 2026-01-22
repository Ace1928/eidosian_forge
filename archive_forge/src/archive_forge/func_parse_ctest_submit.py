from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_submit(ctx, tokens, breakstack):
    """
  ::

    ctest_submit([PARTS <part>...] [FILES <file>...]
                 [SUBMIT_URL <url>]
                 [BUILD_ID <result-var>]
                 [HTTPHEADER <header>]
                 [RETRY_COUNT <count>]
                 [RETRY_DELAY <delay>]
                 [RETURN_VALUE <result-var>]
                 [CAPTURE_CMAKE_ERROR <result-var>]
                 [QUIET]
                 )

  :see: https://cmake.org/cmake/help/latest/command/ctest_submit.html
  """
    kwargs = {'PARTS': PositionalParser('+'), 'FILES': PositionalParser('+'), 'SUBMIT_URL': PositionalParser(1), 'BUILD_ID': PositionalParser(1), 'HTTPHEADER': PositionalParser(1), 'RETRY_COUNT': PositionalParser(1), 'RETRY_DELAY': PositionalParser(1), 'RETURN_VALUE': PositionalParser(1), 'CAPTURE_CMAKE_ERROR': PositionalParser(1)}
    flags = ['QUIET']
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, flags, breakstack)