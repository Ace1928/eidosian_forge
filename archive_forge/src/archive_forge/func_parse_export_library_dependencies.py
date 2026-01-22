from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_export_library_dependencies(ctx, tokens, breakstack):
    """
  ::

    export_library_dependencies(<file> [APPEND])

  :see: https://cmake.org/cmake/help/latest/command/export_library_dependencies.html
  """
    return StandardArgTree.parse(ctx, tokens, '1+', {}, ['APPEND'], breakstack)