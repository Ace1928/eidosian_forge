from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_build(ctx, tokens, breakstack):
    """
  ::

    ctest_build([BUILD <build-dir>] [APPEND]
                [CONFIGURATION <config>]
                [FLAGS <flags>]
                [PROJECT_NAME <project-name>]
                [TARGET <target-name>]
                [NUMBER_ERRORS <num-err-var>]
                [NUMBER_WARNINGS <num-warn-var>]
                [RETURN_VALUE <result-var>]
                [CAPTURE_CMAKE_ERROR <result-var>]
                )

  :see: https://cmake.org/cmake/help/latest/command/ctest_build.html
  """
    kwargs = {'BUILD': PositionalParser(1), 'CONFIGURATION': PositionalParser(1), 'FLAGS': PositionalParser(1), 'PROJECT_NAME': PositionalParser(1), 'TARGET': PositionalParser(1), 'NUMBER_ERRORS': PositionalParser(1), 'NUMBER_WARNINGS': PositionalParser(1), 'RETURN_VALUE': PositionalParser(1), 'CAPTURE_CMAKE_ERROR': PositionalParser(1)}
    return StandardArgTree.parse(ctx, tokens, '+', kwargs, ['APPEND'], breakstack)