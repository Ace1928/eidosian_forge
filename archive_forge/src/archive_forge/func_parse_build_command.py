from cmakelang import lex
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.additional_nodes import (
from cmakelang.parse.util import (
def parse_build_command(ctx, tokens, breakstack):
    """
  ::

    build_command(<variable>
                  [CONFIGURATION <config>]
                  [TARGET <target>]
                  [PROJECT_NAME <projname>] # legacy, causes warning
                 )

  :see: https://cmake.org/cmake/help/latest/command/build_command.html
  """
    kwargs = {'CONFIGURATION': PositionalParser(1), 'TARGET': PositionalParser(1), 'PROJECT_NAME': PositionalParser(1)}
    second_token = get_nth_semantic_token(tokens, 1)
    if second_token is not None and second_token.spelling.upper not in kwargs:
        ctx.lint_ctx.record_lint('W0106', 'build_command(<cachevariable> <makecommand>)', location=tokens[0].get_location())
        return StandardArgTree.parse(ctx, tokens, '2', {}, [], breakstack)
    return StandardArgTree.parse(ctx, tokens, '1', kwargs, [], breakstack)