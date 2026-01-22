from cmakelang.parse.additional_nodes import ShellCommandNode
from cmakelang.parse.argument_nodes import (
def parse_external_project_add_step(ctx, tokens, breakstack):
    """
  ::

    ExternalProject_Add_Step(<name> [<option>...])

  :see: https://cmake.org/cmake/help/v3.14/module/ExternalProject.html#command:externalproject_add_step
  """
    return StandardArgTree.parse(ctx, tokens, npargs=2, kwargs={'COMMAND': ShellCommandNode.parse, 'COMMENT': PositionalParser('+'), 'DEPENDEES': PositionalParser('+'), 'DEPENDERS': PositionalParser('+'), 'DEPENDS': PositionalParser('+'), 'BYPRODUCTS': PositionalParser('+'), 'ALWAYS': PositionalParser(1), 'EXCLUDE_FROM_MAIN': PositionalParser(1), 'WORKING_DIRECTORY': PositionalParser(1), 'LOG': PositionalParser(1), 'USES_TERMINAL': PositionalParser(1)}, flags=[], breakstack=breakstack)