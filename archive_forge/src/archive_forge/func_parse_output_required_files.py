from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_output_required_files(ctx, tokens, breakstack):
    """
  ::

    output_required_files(srcfile outputfile)

  :see: https://cmake.org/cmake/help/latest/command/output_required_files.html
  """
    return StandardArgTree.parse(ctx, tokens, 2, {}, [], breakstack)