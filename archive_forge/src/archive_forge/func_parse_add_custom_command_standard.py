import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.common import KwargBreaker, NodeType, TreeNode
from cmakelang.parse.util import (
def parse_add_custom_command_standard(ctx, tokens, breakstack):
    """
  ::

      add_custom_command(OUTPUT output1 [output2 ...]
                         COMMAND command1 [ARGS] [args1...]
                         [COMMAND command2 [ARGS] [args2...] ...]
                         [MAIN_DEPENDENCY depend]
                         [DEPENDS [depends...]]
                         [BYPRODUCTS [files...]]
                         [IMPLICIT_DEPENDS <lang1> depend1
                                          [<lang2> depend2] ...]
                         [WORKING_DIRECTORY dir]
                         [COMMENT comment]
                         [DEPFILE depfile]
                         [JOB_POOL job_pool]
                         [VERBATIM] [APPEND] [USES_TERMINAL]
                         [COMMAND_EXPAND_LISTS])

  :see: https://cmake.org/cmake/help/latest/command/add_custom_command.html
  """
    subtree = StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={'BYPRODUCTS': PositionalParser('*'), 'COMMAND': ShellCommandNode.parse, 'COMMENT': PositionalParser('*'), 'DEPENDS': PositionalParser('*'), 'DEPFILE': PositionalParser(1), 'JOB_POOL': PositionalParser(1), 'IMPLICIT_DEPENDS': TupleParser(2, '+'), 'MAIN_DEPENDENCY': PositionalParser(1), 'OUTPUT': PositionalParser('+'), 'WORKING_DIRECTORY': PositionalParser(1)}, flags=['APPEND', 'VERBATIM', 'USES_TERMINAL', 'COMMAND_EXPAND_LISTS'], breakstack=breakstack)
    subtree.check_required_kwargs(ctx.lint_ctx, {'COMMAND': 'E1125', 'COMMENT': 'C0113'})
    return subtree