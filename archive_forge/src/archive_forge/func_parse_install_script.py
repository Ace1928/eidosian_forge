import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.util import (
def parse_install_script(ctx, tokens, breakstack):
    """
  ::

    install([[SCRIPT <file>] [CODE <code>]]
            [COMPONENT <component>] [EXCLUDE_FROM_ALL] [...])

  :see: https://cmake.org/cmake/help/v3.14/command/install.html#custom-installation-logic
  """
    return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'SCRIPT': PositionalParser(1), 'CODE': PositionalParser(1), 'COMPONENT': PositionalParser(1)}, flags=['EXCLUDE_FROM_ALL'], breakstack=breakstack)