import argparse
import copy
import re
from tensorflow.python.debug.cli import cli_config
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import source_utils
def list_outputs(self, args, screen_info=None):
    """Command handler for inputs.

    Show inputs to a given node.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    _ = screen_info
    parsed = self._arg_parsers['list_outputs'].parse_args(args)
    output = self._list_inputs_or_outputs(parsed.recursive, parsed.node_name, parsed.depth, parsed.control, parsed.op_type, do_outputs=True)
    node_name = debug_graphs.get_node_name(parsed.node_name)
    _add_main_menu(output, node_name=node_name, enable_list_outputs=False)
    return output