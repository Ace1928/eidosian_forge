from cmakelang.parse.additional_nodes import ShellCommandNode
from cmakelang.parse.argument_nodes import (
def parse_fetchcontent_getproperties(ctx, tokens, breakstack):
    """
  ::

    FetchContent_GetProperties( <name>
      [SOURCE_DIR <srcDirVar>]
      [BINARY_DIR <binDirVar>]
      [POPULATED <doneVar>]
    )
  """
    return StandardArgTree.parse(ctx, tokens, npargs=1, kwargs={'SOURCE_DIR': PositionalParser(1), 'BINARY_DIR': PositionalParser(1), 'POPULATED': PositionalParser(1)}, flags=[], breakstack=breakstack)