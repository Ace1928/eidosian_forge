from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_get_target_property(ctx, tokens, breakstack):
    """
  ::

    get_target_property(VAR target property)

  :see: https://cmake.org/cmake/help/latest/command/get_target_property.html
  """
    return StandardArgTree.parse(ctx, tokens, 3, {}, [], breakstack)