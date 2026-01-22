import argparse
import os
import re
import numpy as np
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import ui_factory
from tensorflow.python.debug.lib import profiling
from tensorflow.python.debug.lib import source_utils
def print_source(self, args, screen_info=None):
    """Print a Python source file with line-level profile information.

    Args:
      args: Command-line arguments, excluding the command prefix, as a list of
        str.
      screen_info: Optional dict input containing screen information such as
        cols.

    Returns:
      Output text lines as a RichTextLines object.
    """
    del screen_info
    parsed = self._arg_parsers['print_source'].parse_args(args)
    device_name_regex = re.compile(parsed.device_name_filter) if parsed.device_name_filter else None
    profile_data = []
    data_generator = self._get_profile_data_generator()
    device_count = len(self._run_metadata.step_stats.dev_stats)
    for index in range(device_count):
        device_stats = self._run_metadata.step_stats.dev_stats[index]
        if device_name_regex and (not device_name_regex.match(device_stats.device)):
            continue
        profile_data.extend(data_generator(device_stats))
    source_annotation = source_utils.annotate_source_against_profile(profile_data, os.path.expanduser(parsed.source_file_path), node_name_filter=parsed.node_name_filter, op_type_filter=parsed.op_type_filter)
    if not source_annotation:
        return debugger_cli_common.RichTextLines(['The source file %s does not contain any profile information for the previous Session run under the following filters:' % parsed.source_file_path, '  --%s: %s' % (_DEVICE_NAME_FILTER_FLAG, parsed.device_name_filter), '  --%s: %s' % (_NODE_NAME_FILTER_FLAG, parsed.node_name_filter), '  --%s: %s' % (_OP_TYPE_FILTER_FLAG, parsed.op_type_filter)])
    max_total_cost = 0
    for line_index in source_annotation:
        total_cost = self._get_total_cost(source_annotation[line_index], parsed.cost_type)
        max_total_cost = max(max_total_cost, total_cost)
    source_lines, line_num_width = source_utils.load_source(parsed.source_file_path)
    cost_bar_max_length = 10
    total_cost_head = parsed.cost_type
    column_widths = {'cost_bar': cost_bar_max_length + 3, 'total_cost': len(total_cost_head) + 3, 'num_nodes_execs': len(self._NUM_EXECS_SUB_HEAD) + 1, 'line_number': line_num_width}
    head = RL(' ' * column_widths['cost_bar'] + total_cost_head + ' ' * (column_widths['total_cost'] - len(total_cost_head)) + self._NUM_NODES_HEAD + ' ' * (column_widths['num_nodes_execs'] - len(self._NUM_NODES_HEAD)), font_attr=self._LINE_COST_ATTR)
    head += RL(self._LINENO_HEAD, font_attr=self._LINE_NUM_ATTR)
    sub_head = RL(' ' * (column_widths['cost_bar'] + column_widths['total_cost']) + self._NUM_EXECS_SUB_HEAD + ' ' * (column_widths['num_nodes_execs'] - len(self._NUM_EXECS_SUB_HEAD)) + ' ' * column_widths['line_number'], font_attr=self._LINE_COST_ATTR)
    sub_head += RL(self._SOURCE_HEAD, font_attr='bold')
    lines = [head, sub_head]
    output_annotations = {}
    for i, line in enumerate(source_lines):
        lineno = i + 1
        if lineno in source_annotation:
            annotation = source_annotation[lineno]
            cost_bar = self._render_normalized_cost_bar(self._get_total_cost(annotation, parsed.cost_type), max_total_cost, cost_bar_max_length)
            annotated_line = cost_bar
            annotated_line += ' ' * (column_widths['cost_bar'] - len(cost_bar))
            total_cost = RL(cli_shared.time_to_readable_str(self._get_total_cost(annotation, parsed.cost_type), force_time_unit=parsed.time_unit), font_attr=self._LINE_COST_ATTR)
            total_cost += ' ' * (column_widths['total_cost'] - len(total_cost))
            annotated_line += total_cost
            file_path_filter = re.escape(parsed.source_file_path) + '$'
            command = 'lp --file_path_filter %s --min_lineno %d --max_lineno %d' % (file_path_filter, lineno, lineno + 1)
            if parsed.device_name_filter:
                command += ' --%s %s' % (_DEVICE_NAME_FILTER_FLAG, parsed.device_name_filter)
            if parsed.node_name_filter:
                command += ' --%s %s' % (_NODE_NAME_FILTER_FLAG, parsed.node_name_filter)
            if parsed.op_type_filter:
                command += ' --%s %s' % (_OP_TYPE_FILTER_FLAG, parsed.op_type_filter)
            menu_item = debugger_cli_common.MenuItem(None, command)
            num_nodes_execs = RL('%d(%d)' % (annotation.node_count, annotation.node_exec_count), font_attr=[self._LINE_COST_ATTR, menu_item])
            num_nodes_execs += ' ' * (column_widths['num_nodes_execs'] - len(num_nodes_execs))
            annotated_line += num_nodes_execs
        else:
            annotated_line = RL(' ' * sum((column_widths[col_name] for col_name in column_widths if col_name != 'line_number')))
        line_num_column = RL(' L%d' % lineno, self._LINE_NUM_ATTR)
        line_num_column += ' ' * (column_widths['line_number'] - len(line_num_column))
        annotated_line += line_num_column
        annotated_line += line
        lines.append(annotated_line)
        if parsed.init_line == lineno:
            output_annotations[debugger_cli_common.INIT_SCROLL_POS_KEY] = len(lines) - 1
    return debugger_cli_common.rich_text_lines_from_rich_line_list(lines, annotations=output_annotations)