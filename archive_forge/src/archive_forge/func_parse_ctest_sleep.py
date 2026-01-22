from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import (
def parse_ctest_sleep(ctx, tokens, breakstack):
    """
  ::

    ctest_sleep(<seconds>)
    ctest_sleep(<time1> <duration> <time2>)

  :see: https://cmake.org/cmake/help/latest/command/ctest_sleep.html
  """
    semantic_tokens = list(iter_semantic_tokens(tokens))
    if len(semantic_tokens) == 1:
        return StandardArgTree.parse(ctx, tokens, 1, {}, [], breakstack)
    return StandardArgTree.parse(ctx, tokens, 3, {}, [], breakstack)