import logging
from cmakelang import lex
from cmakelang.parse.common import KwargBreaker
from cmakelang.parse.simple_nodes import CommentNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.util import (
def parse_install_targets_sub(ctx, tokens, breakstack):
    """
    Parse the inner kwargs of an ``install(TARGETS)`` command. This is common
    logic for ARCHIVE, LIBRARY, RUNTIME, etc.
  :see: https://cmake.org/cmake/help/v3.14/command/install.html#targets
  """
    return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'DESTINATION': PositionalParser(1), 'PERMISSIONS': PositionalParser('+'), 'CONFIGURATIONS': PositionalParser('+'), 'COMPONENT': PositionalParser(1), 'NAMELINK_COMPONENT': PositionalParser(1)}, flags=['OPTIONAL', 'EXCLUDE_FROM_ALL', 'NAMELINK_ONLY', 'NAMELINK_SKIP'], breakstack=breakstack)