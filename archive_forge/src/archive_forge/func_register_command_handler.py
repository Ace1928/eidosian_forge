import argparse
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
def register_command_handler(self, prefix, handler, help_info, prefix_aliases=None):
    """A wrapper around CommandHandlerRegistry.register_command_handler().

    In addition to calling the wrapped register_command_handler() method, this
    method also registers the top-level tab-completion context based on the
    command prefixes and their aliases.

    See the doc string of the wrapped method for more details on the args.

    Args:
      prefix: (str) command prefix.
      handler: (callable) command handler.
      help_info: (str) help information.
      prefix_aliases: (list of str) aliases of the command prefix.
    """
    self._command_handler_registry.register_command_handler(prefix, handler, help_info, prefix_aliases=prefix_aliases)
    self._tab_completion_registry.extend_comp_items('', [prefix])
    if prefix_aliases:
        self._tab_completion_registry.extend_comp_items('', prefix_aliases)