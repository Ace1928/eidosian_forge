import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (
def parse_add_custom_command_events(ctx, tokens, breakstack):
    """
  ::
    add_custom_command(TARGET <target>
                    PRE_BUILD | PRE_LINK | POST_BUILD
                    COMMAND command1 [ARGS] [args1...]
                    [COMMAND command2 [ARGS] [args2...] ...]
                    [BYPRODUCTS [files...]]
                    [WORKING_DIRECTORY dir]
                    [COMMENT comment]
                    [VERBATIM] [USES_TERMINAL]
                    [COMMAND_EXPAND_LISTS])
  :see: https://cmake.org/cmake/help/latest/command/add_custom_command.html
  """
    subtree = StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'BYPRODUCTS': PositionalParser('*'), 'COMMAND': ShellCommandNode.parse, 'COMMENT': PositionalParser('*'), 'TARGET': PositionalParser(1), 'WORKING_DIRECTORY': PositionalParser(1)}, flags=['APPEND', 'VERBATIM', 'USES_TERMINAL', 'COMMAND_EXPAND_LISTS', 'PRE_BUILD', 'PRE_LINK', 'POST_BUILD'], breakstack=breakstack)
    subtree.check_required_kwargs(ctx.lint_ctx, {'COMMAND': 'E1125', 'COMMENT': 'C0113'})
    return subtree