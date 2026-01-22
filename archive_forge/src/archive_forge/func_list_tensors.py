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
def list_tensors(self, args, screen_info=None):
    """Command handler for list_tensors.

    List tensors dumped during debugged Session.run() call.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.

    Raises:
      ValueError: If `--filter_exclude_node_names` is used without `-f` or
        `--tensor_filter` being used.
    """
    _ = screen_info
    parsed = self._arg_parsers['list_tensors'].parse_args(args)
    output = []
    filter_strs = []
    if parsed.op_type_filter:
        op_type_regex = re.compile(parsed.op_type_filter)
        filter_strs.append('Op type regex filter: "%s"' % parsed.op_type_filter)
    else:
        op_type_regex = None
    if parsed.node_name_filter:
        node_name_regex = re.compile(parsed.node_name_filter)
        filter_strs.append('Node name regex filter: "%s"' % parsed.node_name_filter)
    else:
        node_name_regex = None
    output = debugger_cli_common.RichTextLines(filter_strs)
    output.append('')
    if parsed.tensor_filter:
        try:
            filter_callable = self.get_tensor_filter(parsed.tensor_filter)
        except ValueError:
            output = cli_shared.error('There is no tensor filter named "%s".' % parsed.tensor_filter)
            _add_main_menu(output, node_name=None, enable_list_tensors=False)
            return output
        data_to_show = self._debug_dump.find(filter_callable, exclude_node_names=parsed.filter_exclude_node_names)
    else:
        if parsed.filter_exclude_node_names:
            raise ValueError('The flag --filter_exclude_node_names is valid only when the flag -f or --tensor_filter is used.')
        data_to_show = self._debug_dump.dumped_tensor_data
    max_timestamp_width, max_dump_size_width, max_op_type_width = self._measure_tensor_list_column_widths(data_to_show)
    data_to_show = self._sort_dump_data_by(data_to_show, parsed.sort_by, parsed.reverse)
    output.extend(self._tensor_list_column_heads(parsed, max_timestamp_width, max_dump_size_width, max_op_type_width))
    dump_count = 0
    for dump in data_to_show:
        if node_name_regex and (not node_name_regex.match(dump.node_name)):
            continue
        if op_type_regex:
            op_type = self._debug_dump.node_op_type(dump.node_name)
            if not op_type_regex.match(op_type):
                continue
        rel_time = (dump.timestamp - self._debug_dump.t0) / 1000.0
        dump_size_str = cli_shared.bytes_to_readable_str(dump.dump_size_bytes)
        dumped_tensor_name = '%s:%d' % (dump.node_name, dump.output_slot)
        op_type = self._debug_dump.node_op_type(dump.node_name)
        line = '[%.3f]' % rel_time
        line += ' ' * (max_timestamp_width - len(line))
        line += dump_size_str
        line += ' ' * (max_timestamp_width + max_dump_size_width - len(line))
        line += op_type
        line += ' ' * (max_timestamp_width + max_dump_size_width + max_op_type_width - len(line))
        line += dumped_tensor_name
        output.append(line, font_attr_segs=[(len(line) - len(dumped_tensor_name), len(line), debugger_cli_common.MenuItem('', 'pt %s' % dumped_tensor_name))])
        dump_count += 1
    if parsed.tensor_filter:
        output.prepend(['%d dumped tensor(s) passing filter "%s":' % (dump_count, parsed.tensor_filter)])
    else:
        output.prepend(['%d dumped tensor(s):' % dump_count])
    _add_main_menu(output, node_name=None, enable_list_tensors=False)
    return output