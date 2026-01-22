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
def node_info(self, args, screen_info=None):
    """Command handler for node_info.

    Query information about a given node.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    _ = screen_info
    parsed = self._arg_parsers['node_info'].parse_args(args)
    node_name, unused_slot = debug_graphs.parse_node_or_tensor_name(parsed.node_name)
    if not self._debug_dump.node_exists(node_name):
        output = cli_shared.error('There is no node named "%s" in the partition graphs' % node_name)
        _add_main_menu(output, node_name=None, enable_list_tensors=True, enable_node_info=False, enable_list_inputs=False, enable_list_outputs=False)
        return output
    lines = ['Node %s' % node_name]
    font_attr_segs = {0: [(len(lines[-1]) - len(node_name), len(lines[-1]), 'bold')]}
    lines.append('')
    lines.append('  Op: %s' % self._debug_dump.node_op_type(node_name))
    lines.append('  Device: %s' % self._debug_dump.node_device(node_name))
    output = debugger_cli_common.RichTextLines(lines, font_attr_segs=font_attr_segs)
    inputs = self._exclude_denylisted_ops(self._debug_dump.node_inputs(node_name))
    ctrl_inputs = self._exclude_denylisted_ops(self._debug_dump.node_inputs(node_name, is_control=True))
    output.extend(self._format_neighbors('input', inputs, ctrl_inputs))
    recs = self._exclude_denylisted_ops(self._debug_dump.node_recipients(node_name))
    ctrl_recs = self._exclude_denylisted_ops(self._debug_dump.node_recipients(node_name, is_control=True))
    output.extend(self._format_neighbors('recipient', recs, ctrl_recs))
    if parsed.attributes:
        output.extend(self._list_node_attributes(node_name))
    if parsed.dumps:
        output.extend(self._list_node_dumps(node_name))
    if parsed.traceback:
        output.extend(self._render_node_traceback(node_name))
    _add_main_menu(output, node_name=node_name, enable_node_info=False)
    return output