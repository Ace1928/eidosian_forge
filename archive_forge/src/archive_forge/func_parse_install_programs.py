from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_install_programs(ctx, tokens, breakstack):
    """
  ::

    install_programs(<dir> file1 file2 [file3 ...])
    install_programs(<dir> FILES file1 [file2 ...])
    install_programs(<dir> regexp)

  :see: https://cmake.org/cmake/help/latest/command/install_programs.html
  """
    second_token = get_nth_semantic_token(tokens, 1)
    third_token = get_nth_semantic_token(tokens, 2)
    if second_token is not None and second_token.spelling.upper() == 'FILES':
        return StandardArgTree.parse(ctx, tokens, '3+', {}, ['FILES'], breakstack)
    if third_token is None:
        return StandardArgTree.parse(ctx, tokens, 2, {}, [], breakstack)
    return StandardArgTree.parse(ctx, tokens, '3+', {}, [], breakstack)