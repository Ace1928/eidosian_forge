from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_option(ctx, tokens, breakstack):
    """
  ::

    option(<variable> "<help_text>" [value])

  :see: https://cmake.org/cmake/help/latest/command/option.html
  """
    return StandardArgTree.parse(ctx, tokens, '2+', {}, [], breakstack)