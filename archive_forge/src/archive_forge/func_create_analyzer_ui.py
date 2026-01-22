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
def create_analyzer_ui(debug_dump, tensor_filters=None, ui_type='readline', on_ui_exit=None, config=None):
    """Create an instance of ReadlineUI based on a DebugDumpDir object.

  Args:
    debug_dump: (debug_data.DebugDumpDir) The debug dump to use.
    tensor_filters: (dict) A dict mapping tensor filter name (str) to tensor
      filter (Callable).
    ui_type: (str) requested UI type, only "readline" is supported.
    on_ui_exit: (`Callable`) the callback to be called when the UI exits.
    config: A `cli_config.CLIConfig` object.

  Returns:
    (base_ui.BaseUI) A BaseUI subtype object with a set of standard analyzer
      commands and tab-completions registered.
  """
    if config is None:
        config = cli_config.CLIConfig()
    analyzer = DebugAnalyzer(debug_dump, config=config)
    if tensor_filters:
        for tensor_filter_name in tensor_filters:
            analyzer.add_tensor_filter(tensor_filter_name, tensor_filters[tensor_filter_name])
    cli = ui_factory.get_ui(ui_type, on_ui_exit=on_ui_exit, config=config)
    cli.register_command_handler('list_tensors', analyzer.list_tensors, analyzer.get_help('list_tensors'), prefix_aliases=['lt'])
    cli.register_command_handler('node_info', analyzer.node_info, analyzer.get_help('node_info'), prefix_aliases=['ni'])
    cli.register_command_handler('list_inputs', analyzer.list_inputs, analyzer.get_help('list_inputs'), prefix_aliases=['li'])
    cli.register_command_handler('list_outputs', analyzer.list_outputs, analyzer.get_help('list_outputs'), prefix_aliases=['lo'])
    cli.register_command_handler('print_tensor', analyzer.print_tensor, analyzer.get_help('print_tensor'), prefix_aliases=['pt'])
    cli.register_command_handler('print_source', analyzer.print_source, analyzer.get_help('print_source'), prefix_aliases=['ps'])
    cli.register_command_handler('list_source', analyzer.list_source, analyzer.get_help('list_source'), prefix_aliases=['ls'])
    cli.register_command_handler('eval', analyzer.evaluate_expression, analyzer.get_help('eval'), prefix_aliases=['ev'])
    dumped_tensor_names = []
    for datum in debug_dump.dumped_tensor_data:
        dumped_tensor_names.append('%s:%d' % (datum.node_name, datum.output_slot))
    cli.register_tab_comp_context(['print_tensor', 'pt'], dumped_tensor_names)
    return cli