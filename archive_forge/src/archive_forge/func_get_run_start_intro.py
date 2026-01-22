import math
import numpy as np
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
def get_run_start_intro(run_call_count, fetches, feed_dict, tensor_filters, is_callable_runner=False):
    """Generate formatted intro for run-start UI.

  Args:
    run_call_count: (int) Run call counter.
    fetches: Fetches of the `Session.run()` call. See doc of `Session.run()`
      for more details.
    feed_dict: Feeds to the `Session.run()` call. See doc of `Session.run()`
      for more details.
    tensor_filters: (dict) A dict from tensor-filter name to tensor-filter
      callable.
    is_callable_runner: (bool) whether a runner returned by
        Session.make_callable is being run.

  Returns:
    (RichTextLines) Formatted intro message about the `Session.run()` call.
  """
    fetch_lines = common.get_flattened_names(fetches)
    if not feed_dict:
        feed_dict_lines = [debugger_cli_common.RichLine('  (Empty)')]
    else:
        feed_dict_lines = []
        for feed_key in feed_dict:
            feed_key_name = common.get_graph_element_name(feed_key)
            feed_dict_line = debugger_cli_common.RichLine('  ')
            feed_dict_line += debugger_cli_common.RichLine(feed_key_name, debugger_cli_common.MenuItem(None, "pf '%s'" % feed_key_name))
            feed_dict_lines.append(feed_dict_line)
    feed_dict_lines = debugger_cli_common.rich_text_lines_from_rich_line_list(feed_dict_lines)
    out = debugger_cli_common.RichTextLines(_HORIZONTAL_BAR)
    if is_callable_runner:
        out.append('Running a runner returned by Session.make_callable()')
    else:
        out.append('Session.run() call #%d:' % run_call_count)
        out.append('')
        out.append('Fetch(es):')
        out.extend(debugger_cli_common.RichTextLines(['  ' + line for line in fetch_lines]))
        out.append('')
        out.append('Feed dict:')
        out.extend(feed_dict_lines)
    out.append(_HORIZONTAL_BAR)
    out.append('')
    out.append('Select one of the following commands to proceed ---->')
    out.extend(_recommend_command('run', 'Execute the run() call with debug tensor-watching', create_link=True))
    out.extend(_recommend_command('run -n', 'Execute the run() call without debug tensor-watching', create_link=True))
    out.extend(_recommend_command('run -t <T>', 'Execute run() calls (T - 1) times without debugging, then execute run() once more with debugging and drop back to the CLI'))
    out.extend(_recommend_command('run -f <filter_name>', 'Keep executing run() calls until a dumped tensor passes a given, registered filter (conditional breakpoint mode)'))
    more_lines = ['    Registered filter(s):']
    if tensor_filters:
        filter_names = []
        for filter_name in tensor_filters:
            filter_names.append(filter_name)
            command_menu_node = debugger_cli_common.MenuItem('', 'run -f %s' % filter_name)
            more_lines.append(RL('        * ') + RL(filter_name, command_menu_node))
    else:
        more_lines.append('        (None)')
    out.extend(debugger_cli_common.rich_text_lines_from_rich_line_list(more_lines))
    out.append('')
    out.append_rich_line(RL('For more details, see ') + RL('help.', debugger_cli_common.MenuItem('', 'help')) + '.')
    out.append('')
    menu = debugger_cli_common.Menu()
    menu.append(debugger_cli_common.MenuItem('run', 'run'))
    menu.append(debugger_cli_common.MenuItem('exit', 'exit'))
    out.annotations[debugger_cli_common.MAIN_MENU_KEY] = menu
    return out