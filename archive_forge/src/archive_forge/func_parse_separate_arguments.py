from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_separate_arguments(ctx, tokens, breakstack):
    """
  ::

    separate_arguments(<variable> <mode> <args>)

  :see: https://cmake.org/cmake/help/latest/command/separate_arguments.html
  """
    flags = ['UNIX_COMMAND', 'WINDOWS_COMMAND', 'NATIVE_COMMAND']
    return StandardArgTree.parse(ctx, tokens, 3, {}, flags, breakstack)