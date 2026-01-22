from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_fltk_wrap_ui(ctx, tokens, breakstack):
    """
  ::

      fltk_wrap_ui(resultingLibraryName source1
                   source2 ... sourceN )

  :see: https://cmake.org/cmake/help/latest/command/fltk_wrap_ui.html
  """
    return StandardArgTree.parse(ctx, tokens, '2+', {}, [], breakstack)