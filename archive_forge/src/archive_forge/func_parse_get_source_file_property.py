from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_get_source_file_property(ctx, tokens, breakstack):
    """
  ::

    get_source_file_property(VAR file property)

  :see: https://cmake.org/cmake/help/latest/command/get_source_file_property.html
  """
    return StandardArgTree.parse(ctx, tokens, 3, {}, [], breakstack)