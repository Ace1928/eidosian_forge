from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_include_external_msproject(ctx, tokens, breakstack):
    """
  ::

    include_external_msproject(projectname location
                           [TYPE projectTypeGUID]
                           [GUID projectGUID]
                           [PLATFORM platformName]
                           dep1 dep2 ...)

  :see: https://cmake.org/cmake/help/latest/command/include_external_msproject.html
  """
    kwargs = {'TYPE': PositionalParser(1), 'GUID': PositionalParser(1), 'PLATFORM': PositionalParser(1)}
    return StandardArgTree.parse(ctx, tokens, '2+', kwargs, [], breakstack)